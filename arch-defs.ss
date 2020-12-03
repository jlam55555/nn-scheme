#|
A header file of sorts. Define record record types (structs) for neural
network architecture
|#

(define-record-type layer (fields train infer weights))
(define-record-type model (fields layers shape))
