; model training and inference procedures

; training procedure; note that this only trains on single samples
; (as if batch size = 1)
; TODO: add quiet parameter
(define (model-train model lr epochs x y)
  (let epoch-loop ([i 1] [layers (model-layers model)])
    (if (> i epochs)
      (make-model layers (model-shape model))
      [begin
        ; TODO: if not quiet
        ; (display (format "Epoch ~a/~a\n" i epochs))
        (let sub-epoch-loop (
          [x x]
          [y y]
          [layers layers]
        )
          (if (null? x)
            (epoch-loop (1+ i) layers)
            (let* (
              [layers-train-results
                ((layer-train (car layers)) (car x) (car y) lr (cdr layers))]
              [updated-layers (cadr layers-train-results)]
              [loss (caddr layers-train-results)]
            )
              ; TODO: if not quiet
              ; (display (format "Loss: ~a\n" loss))
              (sub-epoch-loop (cdr x) (cdr y) updated-layers)
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
    (lambda (acc infer) (infer acc))
    x
    (map layer-infer (model-layers model))
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
