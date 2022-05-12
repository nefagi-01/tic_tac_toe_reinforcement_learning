import seaborn as sns
import matplotlib.pyplot as plt


def plot_measure(episodes, rewards, parameters, m_opt):
    plt.figure(figsize=(20, 8))
    for i, y_i in enumerate(rewards):
        sns.lineplot(
            x=episodes, y=y_i, label=str(parameters[i])
        )
    plt.title('Mopt (every 250 games) for different values of epsilon' if m_opt else 'Mrand (every 250 games) for '
                                                                                     'different values of epsilon',
              fontsize=20)
    plt.ylabel('Mopt' if m_opt else 'Mrand', fontsize=16)
    plt.xlabel('Episode', fontsize=16)


def plot_average(episodes, rewards, parameters=None):
    plt.figure(figsize=(20, 8))
    if parameters is None:
        sns.lineplot(
            x=episodes, y=rewards
        )
    else:
        for i, y_i in enumerate(rewards):
            sns.lineplot(
                x=episodes, y=y_i, label=str(parameters[i])
            )
    plt.title('Average reward (every 250 games)', fontsize=20)
    plt.ylabel('Average reward', fontsize=16)
    plt.xlabel('Episode', fontsize=16)

    plt.show()
