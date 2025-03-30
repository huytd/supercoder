package com.supercoder

import com.supercoder.ui.TerminalChat
import com.supercoder.agents.CoderAgent
import com.supercoder.config.ArgsParser
import com.supercoder.lib.CursorRulesLoader

object Main {
  def main(args: Array[String]): Unit = {
    ArgsParser.parse(args) match {
      case Some(config) =>
        val additionalPrompt = if config.useCursorRules then CursorRulesLoader.loadRules() else ""
        val modelName = config.model
        val agent = new CoderAgent(additionalPrompt, modelName)
        TerminalChat.run(agent)
      case None =>
        // invalid options, usage error message is already printed by scopt
        sys.exit(1)
    }
  }
}
