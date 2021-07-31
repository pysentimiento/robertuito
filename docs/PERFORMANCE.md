# Pruebas de performance

Hagamos un run de 200 steps

## v2

Acá usamos una n2-standard-16

|Setup         | Batch size  | Iter/s    | Tiempo | MXU   |
|--------------|-------------|-----------|--------|-------|
| Base         |   4k        |  4.38     | 15:06  | ~53%  |
| Base         |   8k        | ~8.5      | 28:53  | ~53%  |
| Dummy-base   |   4k        |  4.08     | 13:35  | ~53%  |
| Dummy-base   |   8k        |  8.00     | 26:28  | ~53%  |



## v3

Acá usamos una n2-standard-16

|Setup         | Batch size  | Iter/s    | Tiempo | MXU   |
|--------------|-------------|-----------|--------|-------|
| Base         |   4k        | 2.77      | 13:02  | 32%   |
| Base         |   8k        |           | 20:13  |~35%   |
| Dummy-base   |   4k        |           | 7:35   |       |
| Dummy-base   |   8k        |           |        |       |

Hay un problema de micros acá...

Si usamos 32 micros

|Setup         | Batch size  | Iter/s    | Tiempo | MXU   |
|--------------|-------------|-----------|--------|-------|
| Base         |   4k        |           |        |       |
| Base         |   8k        |           |        |       |
| Dummy-base   |   4k        |           |        |       |
| Dummy-base   |   8k        |           |        |       |

