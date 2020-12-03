#|
Sigmoid layer implementation. See report for explanation.
|#

(define (sigmoid-layer)
  (let* ([sigmoid (lambda (x) (/ 1 (1+ (exp (- x)))))]
         [sigmoid-prime
          (lambda (x) (let ([sigx (sigmoid x)]) (* sigx (- 1 sigx))))]
         [infer (lambda (x) (map sigmoid x))]
         [train
          (lambda (x y lr nn)
            (let* ([next-layer-results
                    ((layer-train (car nn)) (infer x) y lr (cdr nn))]
                   [next-layer-grads (car next-layer-results)]
                   [next-layers (cadr next-layer-results)]
                   [loss (caddr next-layer-results)]
                   [grads
                    (map
                      (lambda
                        (x next-layer-grad)
                        (* (sigmoid-prime x) next-layer-grad))
                      x next-layer-grads)])
              (list grads (cons (sigmoid-layer) next-layers) loss)))])
    (make-layer train infer '())))
