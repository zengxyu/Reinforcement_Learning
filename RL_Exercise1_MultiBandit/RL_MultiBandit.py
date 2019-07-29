from Bandit import Bandit as Bandit
import numpy as np
import matplotlib.pyplot as plt


def init_bandits(reward_interval_lists):
    # init six bandits with their rewards.
    bandit_list = []
    for i in range(len(reward_interval_lists)):
        b = Bandit(i, reward_interval_lists[i])
        bandit_list.append(b)
    return bandit_list


def get_expected_reward(reward_interval_lists):
    """
    Given is a six-armed bandit, as introduced in the lecture.
    The first arm shall sample its reward uniformly from the interval [1, 3).
    The second arm shall sample its reward uniformly from [-3, 8).
    The third arm shall sample its reward uniformly from the interval [2, 5).
    The fourth arm shall sample its reward uniformly from [–2, 6).
    The fifth arm shall sample its reward uniformly from [3, 4).
    The sixth arm shall sample its reward uniformly from [-2, 2).

    get the expected reward when actions are chosen uniformly
    :return: expected_reward
    """
    expected_reward = 0
    for i in range(len(reward_interval_lists)):
        reward_interval = reward_interval_lists[i]
        mean_value_in_interval = 0.5 * reward_interval[0] + 0.5 * reward_interval[1]
        expected_reward += 1 / 6 * mean_value_in_interval
    return expected_reward


def get_sample_average_reward(time_num):
    # compute the sample average reward for 10 uniformly chosen actions.
    sum_over_rewards = 0
    for i in range(time_num):
        chosen_action = np.random.randint(0, 6)
        reward = bandit_list[chosen_action].get_reward()
        sum_over_rewards += reward
    average_over_rewards = sum_over_rewards / time_num
    return average_over_rewards


def play_once_over_average_reward(sum_rewards_list, times_list, q_list):
    e = 0.1
    d = np.random.random()
    # exploit
    if d < 1 - e:
        # choose the action with high q
        action_chosen = int(np.argmax(q_list))
        reward = bandit_list[action_chosen].get_reward()
        # update the three list
        sum_rewards_list[action_chosen] += reward
        times_list[action_chosen] += 1
        q_list[action_chosen] = sum_rewards_list[action_chosen] / times_list[action_chosen]
    # explore
    else:
        # choose the action randomly
        action_chosen = np.random.randint(0, 6)
        reward = bandit_list[action_chosen].get_reward()
        # update the three list
        sum_rewards_list[action_chosen] += reward
        times_list[action_chosen] += 1
        q_list[action_chosen] = sum_rewards_list[action_chosen] / times_list[action_chosen]


def play_once_over_alpha_lr(times_list, q_list, alpha):
    e = 0.1
    d = np.random.random()
    # exploit
    if d < 1 - e:
        # choose the action with high q
        action_chosen = int(np.argmax(q_list))
        reward = bandit_list[action_chosen].get_reward()
        # update the three list
        times_list[action_chosen] += 1
        # use the alpha learning rate
        q_list[action_chosen] += alpha * (reward - q_list[action_chosen])
    # explore
    else:
        # choose the action randomly
        action_chosen = np.random.randint(0, 6)
        reward = bandit_list[action_chosen].get_reward()
        # update the three list
        times_list[action_chosen] += 1
        # use the alpha learning rate
        q_list[action_chosen] += alpha * (reward - q_list[action_chosen])


def task01():
    print("Task 01 =========================Start")
    expected_reward = get_expected_reward(reward_interval_lists)
    print("expected reward when actions are chosen uniformly: ", expected_reward)
    print("Task 01 =========================End")
    print()


def task02():
    # Implement the six-armed bandit from 1.1) and compute the sample average reward for 10 uniformly chosen actions.
    # Compare this to your expectation from 1.1).
    print("Task 02 =========================Start")
    time_num = 10
    average_over_rewards = get_sample_average_reward(time_num)
    print("the sample average reward for 10 uniformly chosen actions when actions are chosen uniformly: ",
          average_over_rewards)
    print("Task 02 =========================End")
    print()


def task03():
    # e greedy
    # Initialize Q(ai)=0 and chose 4000 actions according to an ε-greedy selection strategy (ε=0.1).
    # Update your action values by computing the sample average reward of each action recursively.
    # For every 100 actions show the percentage of choosing arm 1, arm 2, arm 3, arm 4, arm 5, and arm 6
    # as well as the resulting average reward.
    print("Task 03 =========================Start")
    time_num = 4000
    sum_rewards_list = [0, 0, 0, 0, 0, 0]
    q_list = [0, 0, 0, 0, 0, 0]
    times_list = [0, 0, 0, 0, 0, 0]

    # for record and show
    times_record = []
    percentages_record = [[] for i in range(len(bandit_list))]
    average_rewards_of_all_bandits_record = []

    for i in range(time_num):
        play_once_over_average_reward(sum_rewards_list, times_list, q_list)

        # to record
        times_record.append(i + 1)
        for a in range(len(bandit_list)):
            percentages_record[a].append(times_list[a] / (i + 1))
        average_rewards_of_all_bandits_record.append(np.average(q_list))
    show_results("task03", times_record, percentages_record, average_rewards_of_all_bandits_record)

    print("Task 03 =========================End")
    print()


def task04():
    # e-greedy non stationary
    # Redo the experiment, but after 2000 steps sample the rewards of the fourth arm uniformly from [5, 7).
    # Compare updating action values by computing the sample average reward of each action recursively (as done in 1.3)
    # with using a constant learning rate α=0.01.
    # For every 100 actions show the percentage of choosing arm 1, arm 2, arm 3, arm 4, arm 5, and arm 6
    # as well as the resulting average reward.

    # the learning rate
    alpha = 0.1
    time_num = 10000
    time_step_dynamic = 2000
    print("Task 04 =========================Start")
    print("Compare the two algorithm over average reward and over an alpha learning rate, in non-stationary system")

    # e greedy algorithm over average reward, in non-stationary system
    sum_rewards_list = [0, 0, 0, 0, 0, 0]
    q_list = [0, 0, 0, 0, 0, 0]
    times_list = [0, 0, 0, 0, 0, 0]

    # for record and show
    times_record = []
    percentages_record = [[] for i in range(len(bandit_list))]
    average_rewards_of_all_bandits_record = []

    for i in range(time_num):
        if i == time_step_dynamic:
            bandit_list[3].set_interval([5, 7])
        play_once_over_average_reward(sum_rewards_list, times_list, q_list)

        # to record
        times_record.append(i + 1)
        for a in range(len(bandit_list)):
            percentages_record[a].append(times_list[a] / (i + 1))
        average_rewards_of_all_bandits_record.append(np.average(q_list))

    show_results("task041", times_record, percentages_record, average_rewards_of_all_bandits_record)

    # e greedy algorithm over an alpha learning rate, in non-stationary system
    # don't need sum_rewards_list now, which can save the memory
    bandit_list[3].set_interval([-2, 6])
    q_list = [0, 0, 0, 0, 0, 0]
    times_list = [0, 0, 0, 0, 0, 0]
    # for record and show
    times_record = []
    percentages_record = [[] for i in range(len(bandit_list))]
    average_rewards_of_all_bandits_record = []
    for i in range(time_num):
        # after 2000 steps sample the rewards of the fourth arm uniformly from [5, 7).
        if i == time_step_dynamic:
            bandit_list[3].set_interval([5, 7])
        play_once_over_alpha_lr(times_list, q_list, alpha)

        # to record
        times_record.append(i + 1)
        for a in range(len(bandit_list)):
            percentages_record[a].append(times_list[a] / (i + 1))
        average_rewards_of_all_bandits_record.append(np.average(q_list))
    show_results("task042", times_record, percentages_record, average_rewards_of_all_bandits_record)
    print("Task 04 =========================End")
    print()


def task05():
    # Modify your implementation by using an optimistic initialization Q(ai)=5
    print("Task 05 =========================Start")
    alpha = 0.1
    time_num = 10000
    # at time step 2000, a bandit reward interval changes
    time_step_dynamic = 2000
    q_list = [5, 5, 5, 5, 5, 5]
    times_list = [0, 0, 0, 0, 0, 0]

    # for record and show
    times_record = []
    percentages_record = [[] for i in range(len(bandit_list))]
    average_rewards_of_all_bandits_record = []

    for i in range(time_num):
        # after 2000 steps sample the rewards of the fourth arm uniformly from [5, 7).
        if i == time_step_dynamic:
            bandit_list[3].set_interval([5, 7])
        play_once_over_alpha_lr(times_list, q_list, alpha)

        # to record
        times_record.append(i + 1)
        for a in range(len(bandit_list)):
            percentages_record[a].append(times_list[a] / (i + 1))
        average_rewards_of_all_bandits_record.append(np.average(q_list))

    show_results("task05", times_record, percentages_record, average_rewards_of_all_bandits_record)
    print("Task 05 =========================End")
    print()


def show_results(title, times, percentages, average_rewards_of_all_bandits):
    plt.title(title)
    plt.subplot(211)
    for i in range(len(bandit_list)):
        label_text = "arm" + str(i) + ": " + str(bandit_list[i].get_interval())
        plt.plot(times, percentages[i], label=label_text)
    plt.legend()
    plt.ylabel("percentages")

    plt.subplot(212)
    plt.plot(times, average_rewards_of_all_bandits)
    plt.xlabel("iteration times")
    plt.ylabel("average rewards of all bandits")
    plt.savefig("results/" + title + ".png")
    plt.show()


reward_interval_lists = [[1, 3], [-3, 8], [2, 5], [-2, 6], [3, 4], [-2, 2]]
bandit_list = init_bandits(reward_interval_lists)

# task02()
# task03()
# task04()
task05()
