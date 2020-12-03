#|
Utility functions, mostly for I/O formatting.
|#

; parse input from port as given type
(define (read-as-type port type)
  (let ([input-symbol (read port)])
    (case type
      ['string (symbol->string input-symbol)]
      [else input-symbol]
    )
  )
)

(define (string-trim-front str)
  (let loop ([str (string->list str)])
    (if (or (null? str) (not (char-whitespace? (car str))))
      (list->string str)
      (loop (cdr str))
    )
  )
)

(define (string-join delim lst)
  (fold-left
    (lambda (acc str) (string-append acc delim str))
    (car lst) (cdr lst)
  )
)

; round everything to three decimal places and always have at least
; one digit to the left of the decimal place;
; takes either a list or an atom and applies itself recursively
(define (round-generic places val)
  (if (atom? val)
    (if (zero? places)
      (format "~d" val)
      (string-trim-front
        (format (format "~a~a,~af" #\~ (+ 3 places) places) val)
      )
    )
    (map (lambda (val) (round-generic places val)) val)
  )
)
