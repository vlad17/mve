python3 train_dqn.py $1 1234 & python3 train_dqn.py $1 2345 & python3 train_dqn.py $1 3456 & python3 train_dqn.py $1 4567 &
python3 train_dqn.py $2 1234 & python3 train_dqn.py $2 2345 & python3 train_dqn.py $2 3456 & python3 train_dqn.py $2 4567 &
python3 train_dqn.py $3 1234 & python3 train_dqn.py $3 2345 & python3 train_dqn.py $3 3456 & python3 train_dqn.py $3 4567 && fg
