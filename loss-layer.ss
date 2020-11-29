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
  (make-layer train infer '())
)
