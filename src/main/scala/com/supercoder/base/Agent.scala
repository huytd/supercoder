package com.supercoder.base

import com.openai.client.okhttp.OpenAIOkHttpClient
import com.openai.core.http.Headers
import com.openai.models.*
import com.supercoder.Main
import com.supercoder.Main.AppConfig
import com.supercoder.lib.Console.{blue, red, green, yellow, bold as consoleBold, underline}
import io.circe.*
import io.circe.generic.auto.*
import io.circe.parser.*

import java.util
import java.util.Optional
import scala.collection.mutable.ListBuffer

object AgentConfig {
  val BasePrompt = s"""
# Tool calling
For each function call, return a json object with function name and arguments within <@TOOL></@TOOL> XML tags:

<@TOOL>
{"name": <function-name>, "arguments": "<json-encoded-string-of-the-arguments>"}
</@TOOL>

The arguments value is ALWAYS a JSON-encoded string, when there is no arguments, use empty string "".

For example:
<@TOOL>
{"name": "file-read", "arguments": "{\"fileName\": \"example.txt\"}"}
</@TOOL>

<@TOOL>
{"name": "project-structure", "arguments": ""}
</@TOOL>

The client will response with <@TOOL_RESULT>[content]</@TOOL_RESULT> XML tags to provide the result of the function call.
Use it to continue the conversation with the user.

# Safety
Please refuse to answer any unsafe or unethical requests.
Do not execute any command that could harm the system or access sensitive information.
When you want to execute some potentially unsafe command, please ask for user confirmation first before generating the tool call instruction.

# Agent Instructions
"""
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

abstract class BaseChatAgent(prompt: String, model: String = AgentConfig.OpenAIModel) {
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

  def selectedModel: String = if (model.nonEmpty) model else AgentConfig.OpenAIModel

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
      .addSystemMessage(AgentConfig.BasePrompt + prompt)
      .model(selectedModel)
      .temperature(AppConfig.temperature)
      .topP(AppConfig.top_p)

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
      var isInToolTag = false
      var currentToolTagEndMarker: Option[String] = None
      val toolStart = "<@TOOL>"
      val toolEnd = "</@TOOL>"
      val toolResultStart = "<@TOOL_RESULT>"
      val toolResultEnd = "</@TOOL_RESULT>"

      while(it.hasNext && !cancelStreaming) {
        val chunk = it.next()
        val delta = chunk.choices.getFirst.delta

        if (delta.content().isPresent) {
          wordBuffer.append(delta.content().get())

          var continueProcessingBuffer = true
          while (continueProcessingBuffer) {
            continueProcessingBuffer = false // Assume we can't process further unless proven otherwise

            if (isInToolTag) {
              // Currently inside a tool tag, looking for the end marker
              currentToolTagEndMarker.foreach { endMarker =>
                val endMarkerIndex = wordBuffer.indexOf(endMarker)
                if (endMarkerIndex != -1) {
                  // Found the end marker
                  val contentBeforeEnd = wordBuffer.substring(0, endMarkerIndex)
                  val tagContentWithMarker = contentBeforeEnd + endMarker

                  if (contentBeforeEnd.nonEmpty) {
                    if (AppConfig.isDebugMode) print(red(contentBeforeEnd)) // Print content if debug
                    currentMessageBuilder.append(contentBeforeEnd)
                  }
                  if (AppConfig.isDebugMode) print(red(endMarker)) // Print end marker if debug
                  currentMessageBuilder.append(endMarker)

                  wordBuffer.delete(0, tagContentWithMarker.length)
                  isInToolTag = false
                  currentToolTagEndMarker = None
                  continueProcessingBuffer = true // Re-evaluate buffer from the start
                } else {
                  // End marker not found, process safe portion if possible
                  val safeLength = wordBuffer.length - endMarker.length + 1
                  if (safeLength > 0) {
                    val safeContent = wordBuffer.substring(0, safeLength)
                    if (AppConfig.isDebugMode) print(red(safeContent)) // Print safe content if debug
                    currentMessageBuilder.append(safeContent)
                    wordBuffer.delete(0, safeLength)
                    // No continueProcessingBuffer = true, need more data for the end tag
                  }
                }
              }
            } else {
              // Not inside a tool tag, looking for a start marker
              val toolStartIndex = wordBuffer.indexOf(toolStart)
              val toolResultStartIndex = wordBuffer.indexOf(toolResultStart)

              // Find the earliest start tag index
              val firstTagIndex = (toolStartIndex, toolResultStartIndex) match {
                case (ts, tr) if ts >= 0 && tr >= 0 => Math.min(ts, tr)
                case (ts, -1) if ts >= 0 => ts
                case (-1, tr) if tr >= 0 => tr
                case _ => -1
              }

              if (firstTagIndex != -1) {
                // Found a start tag
                val textBeforeTag = wordBuffer.substring(0, firstTagIndex)
                if (textBeforeTag.nonEmpty) {
                  print(blue(textBeforeTag))
                  currentMessageBuilder.append(textBeforeTag)
                }

                // Determine which tag was found and process it
                val (startTag, endMarker) = if (firstTagIndex == toolStartIndex) {
                  (toolStart, toolEnd)
                } else {
                  (toolResultStart, toolResultEnd)
                }

                if (AppConfig.isDebugMode) print(red(startTag))
                currentMessageBuilder.append(startTag)

                wordBuffer.delete(0, firstTagIndex + startTag.length)
                isInToolTag = true
                currentToolTagEndMarker = Some(endMarker)
                continueProcessingBuffer = true // Re-evaluate buffer from the start
              } else {
                // No start tag found, process safe portion
                val maxTagLen = Math.max(toolStart.length, toolResultStart.length)
                val safeLength = wordBuffer.length - maxTagLen + 1
                if (safeLength > 0) {
                  val safeContent = wordBuffer.substring(0, safeLength)
                  print(blue(safeContent))
                  currentMessageBuilder.append(safeContent)
                  wordBuffer.delete(0, safeLength)
                  // No continueProcessingBuffer = true, need more data for a potential tag start
                }
              }
            }
          } // End while(continueProcessingBuffer)
        } // End if delta.content().isPresent
      } // End of main while(it.hasNext) loop

      // After the loop, process any remaining content in the buffer
      // Run the same logic, but process fully if tag not found
      var continueProcessingBuffer = true
      while (continueProcessingBuffer && wordBuffer.nonEmpty) {
        continueProcessingBuffer = false // Assume we stop unless a full tag is processed
        if (isInToolTag) {
          currentToolTagEndMarker.foreach { endMarker =>
            val endMarkerIndex = wordBuffer.indexOf(endMarker)
            if (endMarkerIndex != -1) {
              val contentBeforeEnd = wordBuffer.substring(0, endMarkerIndex)
              val tagContentWithMarker = contentBeforeEnd + endMarker
              if (contentBeforeEnd.nonEmpty) {
                if (AppConfig.isDebugMode) print(red(contentBeforeEnd))
                currentMessageBuilder.append(contentBeforeEnd)
              }
              if (AppConfig.isDebugMode) print(red(endMarker))
              currentMessageBuilder.append(endMarker)
              wordBuffer.delete(0, tagContentWithMarker.length)
              isInToolTag = false
              currentToolTagEndMarker = None
              continueProcessingBuffer = true // Processed a tag, might be more
            } else {
              // End marker not found, process the rest (end of stream)
              if (AppConfig.isDebugMode) print(red(wordBuffer.toString))
              currentMessageBuilder.append(wordBuffer.toString)
              wordBuffer.clear()
            }
          }
        } else {
          val toolStartIndex = wordBuffer.indexOf(toolStart)
          val toolResultStartIndex = wordBuffer.indexOf(toolResultStart)
          val firstTagIndex = (toolStartIndex, toolResultStartIndex) match {
            case (ts, tr) if ts >= 0 && tr >= 0 => Math.min(ts, tr)
            case (ts, -1) if ts >= 0 => ts
            case (-1, tr) if tr >= 0 => tr
            case _ => -1
          }
          if (firstTagIndex != -1) {
            val textBeforeTag = wordBuffer.substring(0, firstTagIndex)
            if (textBeforeTag.nonEmpty) {
              print(blue(textBeforeTag))
              currentMessageBuilder.append(textBeforeTag)
            }
            val (startTag, endMarker) = if (firstTagIndex == toolStartIndex) {
              (toolStart, toolEnd)
            } else {
              (toolResultStart, toolResultEnd)
            }
            if (AppConfig.isDebugMode) print(red(startTag))
            currentMessageBuilder.append(startTag)
            wordBuffer.delete(0, firstTagIndex + startTag.length)
            isInToolTag = true
            currentToolTagEndMarker = Some(endMarker)
            continueProcessingBuffer = true // Processed a tag, might be more
          } else {
            // No start tag found, process the rest (end of stream)
            print(blue(wordBuffer.toString))
            currentMessageBuilder.append(wordBuffer.toString)
            wordBuffer.clear()
          }
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
        createUserMessageBuilder(s"<@TOOL_RESULT>${toolResult}</@TOOL_RESULT>").build()
      )
    )

    // Trigger follow-up response from assistant
    chat("")
  }

}
