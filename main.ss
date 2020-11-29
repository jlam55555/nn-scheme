(load "arch-defs.ss")
(load "model.ss")
(load "sigmoid-layer.ss")
(load "dense-layer.ss")
(load "loss-layer.ss")
(load "model-io.ss")

; run example
; samples to choose from: wdbc, grades
(define dataset-name "grades")
(define model (load-model (format "weights/~a.init" dataset-name) 3))
(define dataset (load-dataset (format "data/~a.train" dataset-name)))
(define x (car dataset))
(define y (cdr dataset))
(define trained-model (model-train model 0.05 100 x y))
(define test-dataset (load-dataset (format "data/~a.test" dataset-name)))
(display (model-evaluate trained-model (car test-dataset) (cdr test-dataset)))

; prompt-driven model training, returns trained model
(define (prompt-train-model)
  (let* (
    [port (current-input-port)]
    [model-weights-file [begin (display "Model weights: ") (read port)]]
    [dataset-file [begin (display "Train dataset: ") (read port)]]
    [lr [begin (display "Learning rate: ") (read port)]]
    [epochs [begin (display "Epochs: ") (read port)]]
    [out-weights-file [begin (display "Output model weights to: ") (read port)]]
    [model (load-model model-weights-file 3)]
    [dataset (load-dataset dataset-file)]
    [trained-model (model-train model lr epochs (car dataset) (cdr dataset))]
  )
    (export-model-weights model output-file)
    trained-model
  )
)

; prompt-driven model stats (given some trained model)
; TODO: turn dataset into a record type
(define (prompt-test-model model)
  (let* (
    [port (current-input-port)]
    [dataset-file [begin (display "Test dataset: ") (read port)]]
    [out-stats-file [begin (display "Output model stats to: ") (read port)]]
    [dataset (load-dataset dataset-file)]
  )
    (export-model-stats model (car dataset) (cdr dataset) out-stats-file)
  )
)

; a combination of the two
(define (prompt-train-test-model)
  (prompt-test-model (prompt-train-model))
)
