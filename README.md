# tennis_prediction

Tennis prediction project for a university course

Testing different machine learning models algorithm: KNN, Decision trees

## Commands

```bash
python3 -m tennis.py <CMD> # -m runs module as a script
```

## General info

Using randomized CV, instead of grid search, to speed up hyperparams tuning.

## Future updates

- Data transformation: Ordinal encoding for type of tournament
- Feature engineering: Add player ELO
- CLI: improve commands to be more user-friendly
    - predict p1 p2
    - train --model <model_name>
