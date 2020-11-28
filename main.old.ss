; difference of two vectors
(define (vec-diff x1 x2)
  (map - x1 x2)
)

; dot product of two vectors
(define (dot x1 x2)
  (apply + (map * x1 x2))
)

; activation function; hardcoded for now
; sig(x) = 1/(1+e^-x)
(define (sigmoid x)
  (/ 1 (1+ (exp (- x))))
)

; derivative of activation function; hardcoded for now
; dsig(x)/dx = sig(x)*(1-sig(x))
(define (sigmoid-prime x)
  (let ([sigx (sigmoid x)])
    (* sigx (- 1 sigx))
  )
)

; calculate layer output
; for now, hardcode sigmoid as activation
(define (compute-layer-outputs layer inputs)
  ; augment inputs with bias/threshold term
  (let ([inputs (cons -1 inputs)])
    (map
      (lambda (node) (sigmoid (dot node inputs)))
      layer
    )
  )
)

; calculate gradients for each weight in a layer
(define (compute-layer-gradients layer inputs next-layer-grad)
  (map
    (lambda (node next-grad)
      (let ([sig-prime-x (sigmoid-prime (dot node inputs))])
        ; grad_j = sig'(dot(input,weights)) * sum_k{w_{j,k} * grad_k}
        (map
          (lambda (weight) (* sig-prime-x weight next-grad))
          node
        )
      )
    )
    layer next-layer-grad
  )
)

; update each weight via gradient descent; could be combined
; with compute-layer-gradients but separated for clarity
(define (update-layer layer gradients alpha)
  (map
    (lambda (node-weights node-gradients)
      (map
        (lambda (weight gradient) (- weight (* alpha gradient)))
        node-weights node-gradients
      )
    )
    layer gradients
  )
)

; training procedure
; x: vector of inputs
; y: vector of outputs
; alpha: (constant) learning rate
(define (train nn x y alpha)
  (let train-layer (
    [layer nn]
    [inputs x]
  )
    (if (null? layer)

      ; output "sentinel" layer: return gradient (-2*(y-yhat) approx.
      ; (yhat-y)) and empty layer
      (list (list (loss-prime y inputs)) '() (loss y inputs))

      ; other layer: return gradients for each node and updated layers
      ; most of these declarations are for clarity's sake, can be condensed
      (let* (
        [current-layer (car layer)]
        [next-layer (cdr layer)]
        [this-layer-outputs (compute-layer-outputs current-layer inputs)]
        [next-layer-res (train-layer next-layer this-layer-outputs)]
        [next-layer-grad (car next-layer-res)]
        [next-layer-weights (cadr next-layer-weights)]
        [loss (caddr next-layer-weights)]
        [gradients
          (compute-layer-gradients current-layer inputs next-layer-grad)]
        [updated-network (update-layer current-layer gradients alpha)]
      )
        (list gradients updated-network loss)
      )
    )
  )
)

; inference procedure
(define (predict nn x)
  (let* layer-loop (
    [layer nn]
    [inputs x]
  )
    (if (null? layer)

      ; output "sentinel" layer reached; return inferred vector
      inputs

      ; other layer
      (layer-loop
        (cdr layer)
        (calculate-layer-outputs (car layer) inputs)
      )
    )
  )
)
