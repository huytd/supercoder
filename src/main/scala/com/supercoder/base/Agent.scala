package com.supercoder.base

import com.openai.client.okhttp.OpenAIOkHttpClient
import com.openai.core.http.Headers
import com.openai.models.*

import java.util
import java.util.Optional
import scala.collection.mutable.ListBuffer
import com.supercoder.lib.Console.{blue, red}
import io.circe.*
import io.circe.generic.auto.*
import io.circe.parser.*

val BasePrompt = s"""
# Tool calling
For each function call, return a json object with function name and arguments within <@TOOL></@TOOL> XML tags:

<@TOOL>{"name": <function-name>, "arguments": "<json-encoded-string-of-the-arguments>"}</@TOOL>

The arguments value is ALWAYS a JSON-encoded string, when there is no arguments, use empty string "".

For example:
<@TOOL>{"name": "file-read", "arguments": "{\"fileName\": \"example.txt\"}"}</@TOOL>

<@TOOL>{"name": "project-structure", "arguments": ""}</@TOOL>

The client will response with <@TOOL-RESULT>[content]</@TOOL-RESULT> XML tags to provide the result of the function call.
Use it to continue the conversation with the user.

# Response format
When responding to the user, use plain text format. NEVER use Markdown's bold or italic formatting.

# Safety
Please refuse to answer any unsafe or unethical requests.
Do not execute any command that could harm the system or access sensitive information.
When you want to execute some potentially unsafe command, please ask for user confirmation first before generating the tool call instruction.

# Agent Instructions
"""

object AgentConfig {
  val OpenAIAPIBaseURL: String = sys.env.get("SUPERCODER_BASE_URL")
    .orElse(sys.env.get("OPENAI_BASE_URL"))
    .getOrElse("https://api.openai.com/v1")

  val OpenAIModel: String = sys.env.get("SUPERCODER_MODEL")
    .orElse(sys.env.get("OPENAI_MODEL"))
    .getOrElse(ChatModel.O3_MINI.toString)

  val OpenAIAPIKey: String = sys.env.get("SUPERCODER_API_KEY")
    .orElse(sys.env.get("OPENAI_API_KEY"))
    .getOrElse(throw new RuntimeException("You need to config SUPERCODER_API_KEY or OPENAI_API_KEY variable"))
}

case class ToolCallDescription(
    name: String = "",
    arguments: String = "",
) {

  def addName(name: Optional[String]): ToolCallDescription =
    copy(name = this.name + name.orElse(""))

  def addArguments(arguments: Optional[String]): ToolCallDescription =
    copy(arguments = this.arguments + arguments.orElse(""))

}

abstract class BaseChatAgent(prompt: String) {
  private val client = OpenAIOkHttpClient.builder()
    .baseUrl(AgentConfig.OpenAIAPIBaseURL)
    .apiKey(AgentConfig.OpenAIAPIKey)
    .headers(Headers.builder()
      .put("HTTP-Referer", "https://github.com/huytd/supercoder/")
      .put("X-Title", "SuperCoder")
      .build())
    .build()

  private var chatHistory: ListBuffer[ChatCompletionMessageParam] =
    ListBuffer.empty

  def toolExecution(toolCall: ToolCallDescription): String
  def toolDefinitionList: List[FunctionDefinition]

  private def addMessageToHistory(message: ChatCompletionMessageParam): Unit =
    chatHistory = chatHistory :+ message

  private def createAssistantMessageBuilder(
      content: String
  ): ChatCompletionAssistantMessageParam.Builder = {
    ChatCompletionAssistantMessageParam
      .builder()
      .content(content)
      .refusal("")
  }

  private def createUserMessageBuilder(
      content: String
  ): ChatCompletionUserMessageParam.Builder =
    ChatCompletionUserMessageParam
      .builder()
      .content(content)

  // Helper method to build base parameters with system prompt and chat history
  private def buildBaseParams(): ChatCompletionCreateParams.Builder = {
    val params = ChatCompletionCreateParams
      .builder()
      .addSystemMessage(BasePrompt + prompt)
      .model(AgentConfig.OpenAIModel)

    // Add all messages from chat history
    chatHistory.foreach(params.addMessage)
    params
  }

  def chat(message: String): Unit = {
    // Add user message to chat history
    if (message.nonEmpty) {
      addMessageToHistory(
        ChatCompletionMessageParam.ofUser(
          createUserMessageBuilder(message).build()
        )
      )
    }

    val params = buildBaseParams().build()
    val streamResponse = client.chat().completions().createStreaming(params)
    val currentMessageBuilder = new StringBuilder()
    var currentToolCall = ToolCallDescription()

    import sun.misc.{Signal, SignalHandler}
    var cancelStreaming = false
    var streamingStarted = false
    val intSignal = new Signal("INT")
    val oldHandler = Signal.handle(intSignal, new SignalHandler {
      override def handle(sig: Signal): Unit = {
        if (streamingStarted) {
          cancelStreaming = true
        } // else ignore Ctrl+C if streaming hasn't started
      }
    })

    try {
      val it = streamResponse.stream().iterator()
      streamingStarted = true
      val wordBuffer = new StringBuilder()
      var isHiddenTokens = false

      while(it.hasNext && !cancelStreaming) {
        val chunk = it.next()
        val delta = chunk.choices.getFirst.delta

        if (delta.content().isPresent) {
          val chunkContent = delta.content().get()
          currentMessageBuilder.append(chunkContent)
          wordBuffer.append(chunkContent)
          val bufferContent = wordBuffer.toString()
          if (bufferContent.contains(" ")) {
            val words = bufferContent.split(" ")
            val endsWithSpace = bufferContent.last.isWhitespace
            val completeWords = if (endsWithSpace) words else words.dropRight(1)
            for (word <- completeWords) {
              if (word.contains("<@TOOL>")) {
                isHiddenTokens = true
              }
              if (word.contains("</@TOOL>")) {
                isHiddenTokens = false
              }
              if (!isHiddenTokens) {
                print(blue(word + " "))
              }
            }
            wordBuffer.clear()
            if (!endsWithSpace && words.nonEmpty) {
              wordBuffer.append(words.last)
            }
          }
        }
      }

      if (wordBuffer.nonEmpty) {
        val remainingContent = wordBuffer.toString()
        if (remainingContent.nonEmpty) {
          if (!isHiddenTokens) {
            println(blue(remainingContent))
          }
          currentMessageBuilder.append(remainingContent)
        }
      }

      if (cancelStreaming) {
        println(blue("\nStreaming cancelled by user"))
      }
    } catch {
      case e: Exception => e.printStackTrace()
    } finally {
      // Restore original SIGINT handler and close stream
      Signal.handle(intSignal, oldHandler)
      streamResponse.close()
      if (currentMessageBuilder.nonEmpty) {
        println()
        val messageContent = currentMessageBuilder.toString()
        addMessageToHistory(
          ChatCompletionMessageParam.ofAssistant(
            createAssistantMessageBuilder(messageContent)
              .build()
          )
        )

        // Check if the message contains a tool call
        val toolCallRegex = """(?s)<@TOOL>(.*?)</@TOOL>""".r
        val toolCallMatch = toolCallRegex.findFirstMatchIn(messageContent).map(_.group(1))
        if (toolCallMatch.isDefined) {
          val toolCallJson = toolCallMatch.get
          try {
            val parseResult: Either[Error, ToolCallDescription] = decode[ToolCallDescription](toolCallJson)
            currentToolCall = parseResult.getOrElse(ToolCallDescription())
          } catch {
            case e: Exception =>
              println(red(s"Error parsing tool call: ${e.getMessage}"))
          }
        }
      }
      if (currentToolCall.name.nonEmpty) {
        handleToolCall(currentToolCall)
      }
    }
  }

  private def handleToolCall(toolCall: ToolCallDescription): Unit = {
    val toolResult = toolExecution(toolCall)

    // Add the result as assistant's message
    addMessageToHistory(
      ChatCompletionMessageParam.ofAssistant(
        createAssistantMessageBuilder(s"Calling ${toolCall.name} tool...").build()
      )
    )
    addMessageToHistory(
      ChatCompletionMessageParam.ofUser(
        createUserMessageBuilder(s"<@TOOL-RESULT>${toolResult}</@TOOL-RESULT>").build()
      )
    )

    // Trigger follow-up response from assistant
    chat("")
  }

}
