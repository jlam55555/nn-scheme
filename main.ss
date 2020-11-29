(load "arch-defs.ss")
(load "model.ss")
(load "sigmoid-layer.ss")
(load "dense-layer.ss")
(load "loss-layer.ss")
(load "model-io.ss")

; run wdbc example
(define model (load-model "wdbc.init" 3))
(define dataset (load-dataset "wdbc.train"))
(define x (car dataset))
(define y (cdr dataset))
(define trained-model (model-train model 0.1 100 x y))
