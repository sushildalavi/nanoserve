# Scheduler and Batching

## Flow

- request queue
- scheduler
- batch builder
- prefix cache
- model engine
- streaming response

## Emitted metrics

- tokens/sec
- queue depth
- batch size
- TTFT
- TPOT
- prefix cache hit rate

## Notes

- The scheduler is intended to keep Mac-local inference responsive under small to moderate concurrency.

