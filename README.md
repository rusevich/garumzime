# garumzime 
garumzime - transformer-based orthography checker for Latvian

## PLAN:
* data.py - download data, prepare data, serve data
* model.py + config.py - expandable transformer structure for a model
* train.py - training functions
* eval.py - evaluation functions


After this, scripts:
* run_tranining.py - read config, feed to train.py, evaluate with eval.py 
* run_inference.py - use trained model


### Version 0.0.1
```
[0] src : Sodien ir loti skaista diena, un saule spid debesis.
[0] tgt : Šodien ir ļoti skaista diena, un saule spīd debesīs.
[0] pred: Šodie ie ur vatan an nā n pruturatātra stā, vamas ij

[1] src : Es macos latviesu valodu jau piecus gadus.
[1] tgt : Es mācos latviešu valodu jau piecus gadus.
[1] pred: Eiņsasm umu ires pasitībudzevu un vēmu vēc

[2] src : Riga list lietus, un ielas ir slapjas.
[2] tgt : Rīgā līst lietus, un ielas ir slapjas.
[2] pred: Rier pum iemai pākālas pa vis ize vuzm

[3] src : Cels uz mezu ir gars un likumots.
[3] tgt : Ceļš uz mežu ir garš un līkumots.
[3] pred: Ced cas uldis preko ied vāsijošar

[4] src : 2:1 Un tresaja diena Kana, Galileja, bija kazas.
[4] tgt : 2:1 Un trešajā dienā Kānā, Galilejā, bija kāzas.
[4] pred: 240 kapēta notas vieitiru gralveskri māmuzspatīt
```