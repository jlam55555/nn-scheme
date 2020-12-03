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
  (let* ([port (open-input-file filename)]
         ; read in node counts for each layer
         [layer-node-counts
          (let node-counts-loop ([layer-node-counts '()] [i layer-count])
            (if (zero? i)
                (reverse layer-node-counts)
                (node-counts-loop
                  (cons (read port) layer-node-counts) (1- i))))])
    ; create a fc + sigmoid layer for every layer (except input "layer")
    (let layer-loop ([layers '()] [i 0])
      (if (= i (1- layer-count))
          (make-model (reverse (cons (loss-layer) layers)) layer-node-counts)
          (let node-loop ([nodes '()] [j (list-ref layer-node-counts (1+ i))])
            (if (zero? j)
                (layer-loop
                  (cons
                    (sigmoid-layer)
                    (cons (dense-layer (reverse nodes)) layers))
                  (1+ i))
                (let weight-loop ([weights '()]
                                  [k (1+ (list-ref layer-node-counts i))])
                  (if (zero? k)
                    (node-loop (cons (reverse weights) nodes) (1- j))
                    (weight-loop (cons (read port) weights) (1- k))))))))))

; load in dataset; dataset is simply a pair of the x and y matrices
(define (load-dataset filename)
  (let* ([port (open-input-file filename)]
         [N (read port)]
         [input-dim (read port)]
         [output-dim (read port)])
    (let sample-loop ([i N]
                      [x '()]
                      [y '()])
      (if (zero? i)
          ; reversing doesn't matter too much because training is usually
          ; stochastic, but for the sake of this assignment try to make
          ; reproducible so it matches sable's results
          (cons (reverse x) (reverse y))
          (let* ([sample-x
                  (let feature-loop ([j input-dim] [sample-x '()])
                    (if (zero? j)
                        (reverse sample-x)
                        (feature-loop (1- j) (cons (read port) sample-x))))]
                 [sample-y
                  (let output-loop ([j output-dim] [sample-y '()])
                    (if (zero? j)
                        (reverse sample-y)
                        (output-loop (1- j) (cons (read port) sample-y))))])
            (sample-loop (1- i) (cons sample-x x) (cons sample-y y)))))))

; helper; used in export-model-weights, but can be used independently to
; introspect a dense layer
(define (print-layer-weights layer)
  (string-join "\n"
    (map
      (lambda (node) (string-join " " node))
      (round-generic 3 (layer-weights layer)))))

; export model weights to file; note that this makes strict assumptions on the
; network architecture (i.e., FC net with dense first layer)
(define (export-model-weights model filename)
  ; open-output-file throws an error if file exists, so allow overwrite
  ; by deleting an existing file (be careful!)
  (when (file-exists? filename)
        (delete-file filename))
  (let ([port (open-output-file filename)])
    ; export model shape
    (display
      (format "~a\n"
        (string-join " " (map number->string (model-shape model))))
      port)
    ; export layer weights
    (for-each
      (lambda (layer)
        (unless (null? (layer-weights layer))
                (display (format "~a\n" (print-layer-weights layer)) port)))
      (model-layers model))
    ; cleanup; need this to flush output
    (close-output-port port)))

; export model stats to file
(define (export-model-stats model x y filename)
  (when (file-exists? filename)
        (delete-file filename))
  (let* ([port (open-output-file filename)]
         [eval-results (model-evaluate model x y)]
         [contingency-tables (car eval-results)]
         [class-stats (cadr eval-results)]
         [micro-averaged-stats (caddr eval-results)]
         [macro-averaged-stats (cadddr eval-results)])
    (display
      (format "~a\n~a\n~a\n"
        ; class details
        (string-join "\n"
          (map
            (lambda (class) (string-join " " class))
            (map append
              (round-generic 0 contingency-tables)
              (round-generic 3 class-stats))))
        ; micro-averaged stats
        (string-join " " (round-generic 3 micro-averaged-stats))
        ; macro-averaged stats
        (string-join " " (round-generic 3 macro-averaged-stats)))
      port)
    (close-output-port port)))
