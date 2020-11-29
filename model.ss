; model training, inference, and evaluation procedures

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
(define (model-predict model x)
  (fold-left
    (lambda (acc infer) (infer acc))
    x
    (map layer-infer (model-layers model))
  )
)

; apply model on a list of examples
(define (model-predict-set model x)
  (map (lambda (sample) (predict model sample)) x)
)

; binary-output versions of the above
(define (model-binary-predict model x)
  (map (lambda (x) (inexact->exact (round x))) (predict model x))
)
(define (model-binary-predict-set model x)
  (map (lambda (sample) (binary-predict model sample)) x)
)

; generate overall accuracy, precision, recall, f1 given contingency table
(define (contingency-table-stats table)
  (let* (
    [A (car table)]
    [B (cadr table)]
    [C (caddr table)]
    [D (cadddr table)]
    [overall (/ (+ A D) (+ A B C D))]
    [precision (/ A (+ A B))]
    [recall (/ A (+ A C))]
    [f1 (/ (* 2 precision recall) (+ precision recall))]
  )
    (list overall precision recall f1)
  )
)
; generate evalation statistics; assume multiple boolean outputs
(define (model-evaluate model x y)
  (let* (
    ; list of contingency tables
    [contingency-tables
      ; transpose to group by categories rather than samples
      (apply
        map
        ; sum over samples of each category
        (lambda samples-list
          (fold-left
            (lambda (acc sample) (map + acc sample))
            '(0 0 0 0)
            samples-list
          )
        )
        ; count statistics over the samples
        (map
          (lambda (pred est)
            (map
              (lambda (pred est)
                (list
                  (fxlogand pred est)
                  (fxlogand (fx- 1 est) pred)
                  (fxlogand est (fx- 1 pred))
                  (fxlogand (fx- 1 est) (fx- 1 pred))
                )
              )
              pred est
            )
          )
          (model-binary-predict-set model x) y
        )
      )
    ]
    ; calculate per-class stats from contingency tables
    [class-stats (map contingency-table-stats contingency-tables)]
    ; micro averaging sums all of the counts, weights each decision equally
    [micro-averaged-stats
      (contingency-table-stats
        (fold-left
          (lambda (acc table) (map + acc table))
          '(0 0 0 0)
          contingency-tables
        )
      )
    ]
    ; macro averaging averages the counts, weights each class equally
    [macro-averaged-stats
      (map
        (lambda (stat) (/ stat (length class-stats)))
        (fold-left
          (lambda (acc class-stats) (map + acc class-stats))
          '(0 0 0 0)
          class-stats
        )
      )
    ]
  )
    (list
      contingency-tables
      class-stats
      micro-averaged-stats
      macro-averaged-stats
    )
  )
)

; TODO: assertions?