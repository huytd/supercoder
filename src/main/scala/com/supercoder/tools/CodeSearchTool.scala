package com.supercoder.tools

import com.openai.models.FunctionDefinition
import com.supercoder.base.Tool
import com.supercoder.lib.Console.green
import io.circe.*
import io.circe.generic.auto.*
import io.circe.parser.*

import scala.sys.process.*

case class CodeSearchToolArguments(query: String)

object CodeSearchTool extends Tool {

  override val functionDefinition = FunctionDefinition
    .builder()
    .name("code-search")
    .description(
      "Search for code in a given repository. The query parameter should be a regular expression. Arguments: {\"query\": \"<search-query>\"}"
    )
    .build()

  override def execute(arguments: String): String = {
    val parsedArguments = decode[CodeSearchToolArguments](arguments)
    parsedArguments match {
      case Right(args) => {
        val query = args.query
        println(green(s"🔍 Search code for query: ${query}"))
        try {
          val command = s"""rg -e "$query" ."""
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
