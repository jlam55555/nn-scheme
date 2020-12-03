#!/bin/scheme -q
#|
Auto-run the prompted train/test procedures from the command line, without
having to enter the Chez Scheme REPL. With this method, other shell scripting
can be used (e.g., input file redirection to auto-answer the prompts or
timing the whole process). See the report for a more detailed example.

Two ways to use this file:

As an interpreted script (make sure the executable path in the shebang is
correct if Chez Scheme is not located at /bin/scheme):
$ chmod +x autorun.ss && ./autorun.ss

Using the --script argument:
$ scheme -q --script autorun.ss
|#

(load "main.ss")
(prompt-train-test-model)
