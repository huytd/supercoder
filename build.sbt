val scala3Version = "3.6.4"

lazy val root = project
  .in(file("."))
  .enablePlugins(BuildInfoPlugin)
  .settings(
    name := "SuperCoder",
    version := "0.1.4",
    scalaVersion := scala3Version,
    buildInfoKeys := Seq[BuildInfoKey](name, version, scalaVersion, sbtVersion),
    buildInfoPackage := "com.supercoder.build",
    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % "1.0.0" % Test,
      "com.openai" % "openai-java" % "0.33.0",
      "io.circe" %% "circe-core" % "0.14.5",
      "io.circe" %% "circe-generic" % "0.14.5",
      "io.circe" %% "circe-parser" % "0.14.5",
      "org.jline" % "jline" % "3.21.0",
      "com.github.scopt" %% "scopt" % "4.1.0",
      "com.lihaoyi" %% "requests" % "0.8.0",
      "com.lihaoyi" %% "upickle" % "3.1.2"
    )
  )

enablePlugins(JavaAppPackaging)
packageName := "supercoder"
mainClass := Some("com.supercoder.Main")
maintainer := "hey@huy.rocks"
