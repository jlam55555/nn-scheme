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
  (list train infer '())
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

  (list train infer nodes-weights)
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
  (list train infer '())
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
    (map cadr model)
  )
)

; apply model on a list of examples
(define (predict-set model x)
  (map (lambda (sample) (predict model sample)) x)
)

; binary-output versions of the above
(define (binary-predict model x)
  (map round (predict model x))
)
(define (binary-predict-set model x)
  (map (lambda (sample) (binary-predict model sample)) x)
)

; build model; takes in a model built in reverse order, since building in
; reverse is easier to do with one-sided lls; also appends loss layer
; TODO: remove; this is obsoleted by read-fc-netdesc
(define (build-model layers)
  (reverse (cons (loss-layer) layers))
)

; toy example for a single dense perceptron: try to learn z=2x+3y-5
; single perceptron with two inputs, one output
#|
(define model (build-model (list
  (dense-layer '((3 -1 2)))
)))
(define x '((1 5) (2 4) (-2 7) (3 6) (4 2) (-2 3)))
(define y '((12) (11) (12) (19) (9) (0)))
(define m (train model 0.05 1000 x y))
|#


; build fc network using network description as described
; layer count includes input "layer"
; extra parameter layer-count because Sable's network description assumes
; 3 layers, but this can be more general
(define (load-model filename layer-count)
  (let* (
    [port (open-input-file filename)]

    ; read in node counts for each layer
    [layer-node-counts
      (let node-counts-loop ([layer-node-counts '()] [i layer-count])
        (if (zero? i)
          (reverse layer-node-counts)
          (node-counts-loop (cons (read port) layer-node-counts) (1- i))
        )
      )
    ]
  )
    ; create a fc + sigmoid layer for every layer (except input "layer")
    (let layer-loop ([layers '()] [i 0])
      (if (= i (1- layer-count))
        (reverse (cons (loss-layer) layers))
        (let node-loop ([nodes '()] [j (list-ref layer-node-counts (1+ i))])
          (if (zero? j)
            (layer-loop
              (cons (sigmoid-layer) (cons (dense-layer (reverse nodes)) layers))
              (1+ i)
            )
            (let weight-loop ([weights '()] [k (1+ (list-ref layer-node-counts i))])
              (if (zero? k)
                (node-loop (cons (reverse weights) nodes) (1- j))
                (weight-loop (cons (read port) weights) (1- k))
              )
            )
          )
        )
      )
    )
  )
)

; load in dataset
; TODO: add notes on dataset format
(define (load-dataset filename)
  (let* (
    [port (open-input-file filename)]
    [N (read port)]
    [input-dim (read port)]
    [output-dim (read port)]
  )
    (let sample-loop (
      [i N]
      [x '()]
      [y '()]
    )
      (if (zero? i)
        ; reversing doesn't matter too much because training is usually
        ; stochastic, but for the sake of this assignment try to make
        ; reproducible so it matches sable's results
        (cons (reverse x) (reverse y))
        (let* (
          [sample-x
            (let feature-loop ([j input-dim] [sample-x '()])
              (if (zero? j)
                (reverse sample-x)
                (feature-loop (1- j) (cons (read port) sample-x))
              )
            )
          ]
          [sample-y
            (let output-loop ([j output-dim] [sample-y '()])
              (if (zero? j)
                (reverse sample-y)
                (output-loop (1- j) (cons (read port) sample-y))
              )
            )
          ]
        )
          (sample-loop (1- i) (cons sample-x x) (cons sample-y y))
        )
      )
    )
  )
)

; export model
; TODO: working here

; run wdbc example
(define model (load-model "wdbc.init" 3))
(define dataset (load-dataset "wdbc.train"))
(define x (car dataset))
(define y (cdr dataset))
(define trained-model (train model 0.1 100 x y))

; for displaying layer weights (for dense layer)
(define (lw layer)
  (map
    (lambda (x) (map (lambda (x) (/ (round (* x 1000)) 1000)) x))
    (caddr layer)
  )
)

(display (lw (car trained-model)))
(display (lw (caddr trained-model)))
