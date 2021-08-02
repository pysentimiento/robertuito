# Pruebas de performance

Hagamos un run de 200 steps

## v2

Acá usamos una n2-standard-16

|Setup         | Batch size  | Iter/s    | Tiempo | MXU   |
|--------------|-------------|-----------|--------|-------|
| Base         |   4k        |  4.5      | 15:06  | ~53%  |
| Base         |   8k        | ~8.5      | 28:53  | ~53%  |
| Dummy-base   |   4k        |  4.08     | 13:35  | ~53%  |
| Dummy-base   |   8k        |  8.00     | 26:28  | ~53%  |


Entonces, 400k por 4k toma =>

(400k * 4.5s) (3600s)

## 2x1080Ti

|Setup         | Batch size  | s/Iter    | Tiempo | MXU   |
|--------------|-------------|-----------|--------|-------|
| Base         |   4k        | ~16s/i    | 50:00  |       |
| Base         |   8k        | ~28s/i    |        |       |
| Dummy-base   |   4k        | 26.62     | 100:00 |       |
| Dummy-base   |   8k        |           |        |       |

Una cosita nomás: acá usamos padding no a longitud máxima, aprovechando que las GPU pueden hacer algo más dinámico

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



### Scripts

#### Grito

```bash
deepspeed --num_gpus 2 bin/run_mlm.py config/performance_test/4k-grito-dummy.json
deepspeed --num_gpus 2 bin/run_mlm.py config/performance_test/4k-grito.json
```

#### v2
