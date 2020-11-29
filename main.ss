(load "arch-defs.ss")
(load "model.ss")
(load "sigmoid-layer.ss")
(load "dense-layer.ss")
(load "loss-layer.ss")
(load "model-io.ss")

; run example
; samples to choose from: wdbc, grades
; TODO: remove this or move to another file
#|
(define dataset-name "grades")
(define model (load-model (format "weights/~a.init" dataset-name) 3))
(define dataset (load-dataset (format "data/~a.train" dataset-name)))
(define x (car dataset))
(define y (cdr dataset))
(define trained-model (model-train model 0.05 100 x y))
(define test-dataset (load-dataset (format "data/~a.test" dataset-name)))
(define x (car test-dataset))
(define y (cdr test-dataset))
|#

; parse input as type
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
