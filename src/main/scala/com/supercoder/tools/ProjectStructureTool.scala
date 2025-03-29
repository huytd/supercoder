package com.supercoder.tools

import com.openai.models.FunctionDefinition
import com.supercoder.base.Tool
import com.supercoder.lib.Console.green

import scala.sys.process.*

object ProjectStructureTool extends Tool {

  override val functionDefinition = FunctionDefinition
    .builder()
    .name("project-structure")
    .description("Get the structure of the current project. Arguments: 'null'")
    .build()

  override def execute(arguments: String): String = {
    println(green("🔎 Reading project structure..."))
    try {
      val command = "tree --gitignore"
      command.!!
    } catch {
      case e: Exception => s"Error getting project structure: ${e.getMessage}"
    }
  }

}
