package com.supercoder.config

import scopt.OParser

case class Config(useCursorRules: Boolean = false, model: String = "", isDebugMode: Boolean = false)

object ArgsParser {
  def parse(args: Array[String]): Option[Config] = {
    val builder = OParser.builder[Config]
    val parser = {
      import builder._
      OParser.sequence(
        programName("SuperCoder"),
        opt[String]('c', "use-cursor-rules")
          .action((x, c) => c.copy(useCursorRules = (x == "true")))
          .text("use Cursor rules for the agent"),
        opt[String]('m', "model")
          .action((x, c) => c.copy(model = x))
          .text("model to use for the agent"),
        opt[String]('d', "debug")
          .action((x, c) => c.copy(isDebugMode = (x == "true")))
          .text("enable debug mode"),
        help("help").text("prints this usage text")
      )
    }
    OParser.parse(parser, args, Config())
  }
}
