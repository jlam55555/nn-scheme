; technically this doesn't have to be a function, since there are no weights/state;
; use function just to be consistent with other layer types (e.g., dense), which do store state
(define (sigmoid-layer)
  (define (sigmoid x) (/ 1 (1+ (exp (- x)))))
  (define (sigmoid-prime x)
    (let ([sigx (sigmoid x)]) (* sigx (- 1 sigx))))

  (define (train x y lr nn)
    (let* (
      [next-layer-results ((layer-train (car nn)) (infer x) y lr (cdr nn))]
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
  (make-layer train infer '())
)
