package com.supercoder.tools

import com.openai.models.FunctionDefinition
import com.supercoder.base.Tool
import com.supercoder.lib.Console.green
import io.circe.*
import io.circe.generic.auto.*
import io.circe.parser.*

import java.io.{File, PrintWriter}

case class CodeEditToolArguments(filepath: String, content: String)

object CodeEditTool extends Tool {

  override val functionDefinition = FunctionDefinition
    .builder()
    .name("code-edit")
    .description(
      "Edit a code file in the repository. Provide the file path and the new content for the file. Arguments: {\"filepath\": \"<file-path>\", \"content\": \"<new-content>\"}"
    )
    .build()

  override def execute(arguments: String): String = {
    val parsedArguments = decode[CodeEditToolArguments](arguments)
    parsedArguments match {
      case Right(args) => {
        val filepath = args.filepath
        val content = args.content
        println(green(s"✏️ Editing file: ${filepath}"))

        try {
          val file = new File(filepath)
          // Create directory if it doesn't exist
          file.getParentFile.mkdirs()

          // Write the content to the file
          val writer = new PrintWriter(file)
          writer.write(content)
          writer.close()

          s"Successfully edited file: ${filepath}"
        } catch {
          case e: Exception => s"Error editing file: ${e.getMessage}"
        }
      }
      case Left(error) => s"Error: Invalid arguments - ${error.getMessage}"
    }
  }
}
