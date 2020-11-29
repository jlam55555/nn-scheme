; dense layer stores weights, needs to be a function
(define (dense-layer nodes-weights)
  (define (train x y lr nn)
    (let* (
      [next-layer-results ((layer-train (car nn)) (infer x) y lr (cdr nn))]
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

  (make-layer train infer nodes-weights)
)
