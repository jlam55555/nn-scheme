#|
Main entry point: loads dependencies and defines procedures to automate
the train/test procedures by prompting for the necessary inputs.

An alternative entry point is autorun.ss, which simply loads this file and
calls (prompt-train-test-model) (and thus can be run from the command line).
|#

(load "arch-defs.ss")
(load "model.ss")
(load "sigmoid-layer.ss")
(load "dense-layer.ss")
(load "loss-layer.ss")
(load "model-io.ss")

; helper function parse input as given type
(define (read-as-type port type)
  (let ([input-symbol (read port)])
    (case type
      ['string (symbol->string input-symbol)]
      [else input-symbol]
    )
  )
)

; prompt-driven model training, returns trained model
(define (prompt-train-model)
  (let* (
    [port (current-input-port)]
    [model-weights-file [begin
      (display "Model weights file: ")
      (read-as-type port 'string)
    ]]
    [dataset-file [begin
      (display "Train dataset file: ")
      (read-as-type port 'string)
    ]]
    [lr [begin
      (display "Learning rate: ")
      (read-as-type port 'number)
    ]]
    [epochs [begin
      (display "Epochs: ")
      (read-as-type port 'number)
    ]]
    [out-weights-file [begin
      (display "Output model weights to: ")
      (read-as-type port 'string)
    ]]
    [model (load-model model-weights-file 3)]
    [dataset (load-dataset dataset-file)]
    [trained-model (model-train model lr epochs (car dataset) (cdr dataset))]
  )
    (export-model-weights trained-model out-weights-file)
    trained-model
  )
)

; prompt-driven model stats (given some trained model)
; TODO: turn dataset into a record type
(define (prompt-test-model model)
  (let* (
    [port (current-input-port)]
    [dataset-file [begin
      (display "Test dataset file: ")
      (read-as-type port 'string)
    ]]
    [out-stats-file [begin
      (display "Output model stats to: ")
      (read-as-type port 'string)
    ]]
    [dataset (load-dataset dataset-file)]
  )
    (export-model-stats model (car dataset) (cdr dataset) out-stats-file)
  )
)

; a combination of the two
(define (prompt-train-test-model)
  (prompt-test-model (prompt-train-model))
)
