#|
Procedures to read and write to the file formats as described in the
report. Here are the file types and associated methods

1. Dataset files (either train or test)
- load-dataset -> reads dataset files into dataset instance

2. Weights files
- load-model -> reads model weights into model instance
- export-model-weights -> outputs model weights file

3. Statistics files
- export-model-stats -> outputs statistics file
|#

; build fc network using network description as described; layer count includes
; input "layer"; extra parameter layer-count is always 3 for the sake of this
; assignment (one hidden layer), but can be generalized to more layers
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
        (make-model
          (reverse (cons (loss-layer) layers))
          layer-node-counts
        )
        (let node-loop ([nodes '()] [j (list-ref layer-node-counts (1+ i))])
          (if (zero? j)
            (layer-loop
              (cons (sigmoid-layer) (cons (dense-layer (reverse nodes)) layers))
              (1+ i)
            )
            (let weight-loop (
              [weights '()]
              [k (1+ (list-ref layer-node-counts i))]
            )
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

; for displaying layer weights (for dense layer) to three decimal places
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
(define (print-layer-weights layer)
  (string-join "\n"
    (map
      (lambda (x)
        (string-join " "
          (map (lambda (x) (string-trim-front (format "~6,3f" x))) x)
        )
      )
      (layer-weights layer)
    )
  )
)

; export model weights to file; note that this makes strict assumptions on the
; network architecture (i.e., FC net with dense first layer)
; TODO: rename to model-export and model-load...?
; TODO: if no filename, then print to stdout?
(define (export-model-weights model filename)
  ; open-output-file throws an error if file exists, so allow overwrite
  ; by deleting an existing file (be careful!)
  (when (file-exists? filename)
    (delete-file filename)
  )
  (let ([port (open-output-file filename)])
    ; export model shape
    (display
      (format "~a\n"
        (string-join " " (map number->string (model-shape model)))
      )
      port
    )
    ; export layer weights
    (for-each
      (lambda (layer)
        (unless (null? (layer-weights layer))
          (display (format "~a\n" (print-layer-weights layer)) port)
        )
      )
      (model-layers model)
    )
    ; cleanup; need this to flush output
    (close-output-port port)
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

; export model stats to file
(define (export-model-stats model x y filename)
  (when (file-exists? filename)
    (delete-file filename)
  )
  (let* (
    [port (open-output-file filename)]
    [eval-results (model-evaluate model x y)]
    [contingency-tables (car eval-results)]
    [class-stats (cadr eval-results)]
    [micro-averaged-stats (caddr eval-results)]
    [macro-averaged-stats (cadddr eval-results)]
  )
    (display
      (format "~a\n~a\n~a\n"
        ; class details
        (string-join "\n"
          (map
            (lambda (class) (string-join " " class))
            (map append
              (round-generic 0 contingency-tables)
              (round-generic 3 class-stats)
            )
          )
        )
        ; micro-averaged stats
        (string-join " " (round-generic 3 micro-averaged-stats))
        ; macro-averaged stats
        (string-join " " (round-generic 3 macro-averaged-stats))
      )
      port
    )
    (close-output-port port)
  )
)
