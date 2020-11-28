; each layer is a list of three elements: train method, inference method, and weights
; train method takes the current input and the rest of the neural net, and returns gradients
; infer method performs calculation and returns result

; technically this doesn't have to be a function, since there are no weights/state;
; use function just to be consistent with other layer types (e.g., dense), which do store state
(define (sigmoid-layer)
  (define (sigmoid x) (/ 1 (1+ (exp (- x)))))
  (define (sigmoid-prime x)
    (let ([sigx (sigmoid x)]) (* sigx (- 1 sigx))))

  (define (train x y lr nn)
    (let* (
      [next-layer-results ((car (car nn)) (infer x) y lr (cdr nn))]
      [next-layer-grads (car next-layer-results)]
      [next-layers (cadr next-layer-results)]
      [loss (caddr next-layer-results)]
      [grads
        (map
          (lambda (x next-layer-grad) (* (sigmoid-prime x) next-layer-grad))
          x next-layer-grads
        )
      ]
    )
      (list grads (cons (sigmoid-layer) next-layers) loss)
    )
  )
  (define (infer x) (map sigmoid x))
  (cons train infer)
)

; dense layer stores weights, needs to be a function
(define (dense-layer nodes-weights)
  (define (train x y lr nn)
    (let* (
      [next-layer-results ((car (car nn)) (infer x) y lr (cdr nn))]
      [next-layer-grads (car next-layer-results)]
      [next-layers (cadr next-layer-results)]
      [loss (caddr next-layer-results)]
      [updated-weights
        ; augment x again
        (let ([x (cons -1 x)])
          (map
            (lambda (output-node-weights output-node-grad)
              (map
                (lambda (weight input)
                  ; update rule: w <- w - lr * grad(loss)_w
                  (- weight (* lr input output-node-grad))
                )
                output-node-weights x
              )
            )
            nodes-weights next-layer-grads
          )
        )
      ]
      [grads
        (map
          (lambda (input-node-weights)
            (apply + (map * input-node-weights next-layer-grads))
          )
          (cdr (apply map list nodes-weights))
        )
      ]
    )
      (list grads (cons (dense-layer updated-weights) next-layers) loss)
    )
  )
  (define (infer x)
    ; augment x
    (let ([x (cons -1 x)])
      (map
        (lambda (node-weights)
          (apply + (map * node-weights x))
        )
        nodes-weights
      )
    )
  )
  (cons train infer)
)

; "loss layer" handles loss during training and passes inputs through
; during inference; right now loss is hardcoded as MSE
(define (loss-layer)
  ; MSE loss is sum of (x-y).^2
  ; gradient of MSE loss (w.r.t. x) is 2*(x-y) (can drop constant factor)
  (define (train x y lr _)
    (let* (
      [grads (map - x y)]
      [loss (apply + (map (lambda (x y) (expt (- x y) 2)) x y))]
    )
      (list grads (list (loss-layer)) loss)
    )
  )
  (define (infer x) x)
  (cons train infer)
)

; training procedure; uses a batch-size of 1 because I am too lazy to implement
; the backprop generally enough to include matrix ops;
; x and y are matrices (lists of lists) of the training data
(define (train model lr epochs x y)
  (let epoch-loop ([i 1] [model model])
    (if (> i epochs)
      model
      [begin
        (display (format "Epoch ~a/~a\n" i epochs))
        (let sub-epoch-loop (
          [x x]
          [y y]
          [model model]
        )
          (if (null? x)
            (epoch-loop (1+ i) model)
            (let* (
              [train-method (car (car model))]
              [next-layer (cdr model)]
              [model-train-results (train-method (car x) (car y) lr next-layer)]
              [updated-model (cadr model-train-results)]
              [loss (caddr model-train-results)]
            )
              (display (format "Loss: ~a\n" loss))
              (sub-epoch-loop (cdr x) (cdr y) updated-model)
            )
          )
        )
      ]
    )
  )
)

; apply model on single sample
(define (predict model x)
  (fold-left
    (lambda (accumulator infer-method) (infer-method accumulator))
    x
    (map
      (lambda (layer) (cdr layer))
      model
    )
  )
)

; apply model on list of samples


; build model; takes in a model built in reverse order, since building in
; reverse is easier to do with one-sided lls; also appends loss layer
(define (build-model layers)
  (reverse (cons (loss-layer) layers))
)

; toy example: try to learn z=2x+3y-5
; single perceptron with two inputs, one output
(define model (build-model (list
  (dense-layer '((3 -1 2)))
)))
(define x '((1 5) (2 4) (-2 7) (3 6) (4 2) (-2 3)))
(define y '((12) (11) (12) (19) (9) (0)))
(define m (train model 0.05 1000 x y))
