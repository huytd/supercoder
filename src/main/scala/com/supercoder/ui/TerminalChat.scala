package com.supercoder.ui

import com.supercoder.base.BaseChatAgent
import com.supercoder.lib.Console
import com.supercoder.lib.Console.{blue, bold, green, underline}
import com.supercoder.build.BuildInfo
import org.jline.reader.{LineReader, LineReaderBuilder, Reference, Widget}
import org.jline.terminal.{Terminal, TerminalBuilder}

object TerminalChat {

  def clearScreen(): Unit = {
    print("\u001b[2J")
    print("\u001b[H")
  }

  def printHeader(agent: BaseChatAgent): Unit = {
    clearScreen()
    println(blue("█▀ █░█ █▀█ █▀▀ █▀█ █▀▀ █▀█ █▀▄ █▀▀ █▀█"))
    println(blue("▄█ █▄█ █▀▀ ██▄ █▀▄ █▄▄ █▄█ █▄▀ ██▄ █▀▄"))
    println(blue(s"v${BuildInfo.version}"))
    println()
    println(blue(s"Model: ${agent.selectedModel}"))
    println(blue("Type '/help' for available commands.\n"))
  }

  def showHelp(): Unit = {
    println(underline("Available commands:"))
    println(s"  ${bold("/help")}  - Display this help message")
    println(s"  ${bold("/clear")} - Clear the terminal screen")
    println(s"  ${bold("exit")}\t- Terminate the chat session")
    println(s"  ${bold("bye")}\t- Terminate the chat session\n")
    println("Just type any message to chat with the agent.")
    println("To insert a new line in your message, use Shift+Enter.")
  }

  def run(agent: BaseChatAgent): Unit = {
    printHeader(agent)
    val terminal: Terminal = TerminalBuilder.builder().system(true).build()
    val reader: LineReader = LineReaderBuilder.builder().terminal(terminal).build()

    // Add a widget to insert a newline when Shift+Enter is pressed.
    // Note: The escape sequence for Shift+Enter can vary between terminals; 
    // here we assume "\u001b[13;2u" is the sequence for Shift+Enter.
    reader.getWidgets.put("insert-newline", new Widget {
      override def apply(): Boolean = {
        // Insert a newline character into the current buffer
        reader.getBuffer.write("\n")
        // Refresh display to show the new line in the prompt
        reader.callWidget(LineReader.REDRAW_LINE)
        reader.callWidget(LineReader.REDISPLAY)
        true
      }
    })

    // Bind the Shift+Enter key sequence to our widget.
    // The escape sequence here (\u001b[13;2u) might need adjustment based on your terminal emulator.
    val mainKeyMap = reader.getKeyMaps.get(LineReader.MAIN)
    mainKeyMap.bind(new Reference("insert-newline"), "\u001b[13;2u")

    var keepRunning = true
    while (keepRunning) {
      try {
        val input = reader.readLine(bold("> "))
        if (input == null) {
          keepRunning = false
        } else {
          input.trim match {
            case "" => // ignore empty input
            case "/help" => showHelp()
            case "/clear" =>
              clearScreen()
              printHeader(agent)
            case "exit" | "bye" =>
              println(blue("\nChat session terminated. Goodbye!"))
              keepRunning = false
            case message =>
              agent.chat(message)
          }
        }
      } catch {
        case _: org.jline.reader.UserInterruptException => // Handle ctrl+C gracefully
          println(blue("\nChat session terminated. Goodbye!"))
          keepRunning = false
        case e: Exception => e.printStackTrace()
      }
    }
  }
}