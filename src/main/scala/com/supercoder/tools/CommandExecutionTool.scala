package com.supercoder.tools

import com.openai.models.FunctionDefinition
import com.supercoder.base.Tool
import com.supercoder.lib.Console.green
import io.circe.*
import io.circe.generic.auto.*
import io.circe.parser.*

import scala.sys.process.*

case class CommandExecutionToolArguments(command: String)

object CommandExecutionTool extends Tool {

  override val functionDefinition = FunctionDefinition
    .builder()
    .name("command-execution")
    .description(
      "Execute a shell command on the user's terminal, and pass the output back to the agent. Arguments: {\"command\": \"<shell-command>\"}"
    )
    .build()

  override def execute(arguments: String): String = {
    val parsedArguments = decode[CommandExecutionToolArguments](arguments)
    parsedArguments match {
      case Right(args) => {
        val command = args.command
        println(green(s"🔍 Execute shell command: ${command}"))
        try {
          val output: String = command.!!
          return output
        } catch {
          case e: Exception => s"Error: ${e.getMessage}"
        }
      }
      case _ => "Error: Invalid arguments"
    }
  }

}
