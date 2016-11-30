import numpy as np

np.random.seed(1337)


def load_story_cloze(name):
    story = []
    order = []

    with open("./quebap/data/StoryCloze/%s.tsv" % name, "r") as f:
        for line in f.readlines()[1:]:
            current_order = np.random.permutation(5)

            if name == "train":
                sha, title, s1, s2, s3, s4, s5 = \
                    [x.strip() for x in line.split("\t")]
            else:
                sha, s1, s2, s3, s4, q1, q2, label = \
                    [x.strip() for x in line.split("\t")]
                if label == "1":
                    s5 = q1
                else:
                    s5 = q2

            sorted_story = [s1, s2, s3, s4, s5]

            current_story = []
            for i in current_order:
                current_story.append(sorted_story[i])

            story.append(current_story)
            order.append(current_order)
    f.close()
    return story, order

for corpus in ["train", "dev", "test"]:
    story, order = load_story_cloze(corpus)
    # for i in range(len(context)):
    print(corpus, len(story), len(order))
    for i in range(3):
        print(story[i], order[i])

    with open("./quebap/data/StoryCloze/%s_shuffled.tsv" % corpus, "w") as f:
        for i in range(len(story)):
            f.write("%s\t%s\n" % ("\t".join(story[i]),
                                  "\t".join([str(x) for x in order[i]])))
        f.close()

