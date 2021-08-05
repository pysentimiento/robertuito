# Pruebas de performance

Hagamos un run de 200 steps

## v2

Acá usamos una n2-standard-16

|Setup         | Batch size  | Iter/s    | Tiempo | MXU   |
|--------------|-------------|-----------|--------|-------|
| Base         |   4k        |  4.3      | 15:06  | ~53%  |
| Base         |   8k        | ~8.5      | 28:53  | ~53%  |
| Dummy-base   |   4k        |  4.08     | 13:35  | ~53%  |
| Dummy-base   |   8k        |  8.00     | 26:28  | ~53%  |


Entonces, 400k por 4k toma =>

(100k * 4.3s) (3600s)

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
| Base         |   4k        | ~2.66     | 13:02  | 32%   |
| Base         |   8k        |           | 20:13  |~35%   |
| Dummy-base   |   4k        |~2.2       | 7:35   |       |
| Dummy-base   |   8k        |           |        |       |


Hay un problema de micros acá...

Si usamos 32 micros

|Setup         | Batch size  | Iter/s    | Tiempo | MXU   |
|--------------|-------------|-----------|--------|-------|
| Base         |   4k        | 2.24      |  8:38  |       |
| Base         |   8k        | ~5        |  16:48 |       |
| Dummy-base   |   4k        | 2.1       |  8:19  |       |
| Dummy-base   |   8k        |           |        |       |

Acá un bsz de 4k va!

2.5s por step son  1440 por hora, ~34k steps por días. 200k steps tomaría alrededor de 6 días. Vamos con esa


### Scripts

#### Grito

```bash
deepspeed --num_gpus 2 bin/run_mlm.py config/performance_test/4k-grito-dummy.json
deepspeed --num_gpus 2 bin/run_mlm.py config/performance_test/4k-grito.json
```

#### v2

```bash
python bin/xla_spawn.py --num_cores 8 bin/run_mlm.py config/performance_test/4k-v2.json
```
