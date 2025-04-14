package com.supercoder.config

import scopt.OParser

case class Config(
  useCursorRules: Boolean = false,
  model: String = "",
  isDebugMode: Boolean = false,
  temperature: Double = 0.2,
  top_p: Double = 0.1
)

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
        opt[Double]('t', "temperature")
          .action((x, c) => c.copy(temperature = x))
          .text("temperature for LLM calls (default: 0.2)"),
        opt[Double]('p', "top_p")
          .action((x, c) => c.copy(top_p = x))
          .text("top_p for LLM nucleus sampling (default: 0.1)"),
        help("help").text("prints this usage text")
      )
    }
    OParser.parse(parser, args, Config())
  }
}
