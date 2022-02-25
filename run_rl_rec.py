from tabulate import tabulate
import inquirer
import torch as t
import pickle
import numpy as np

# This is a 3 layer neural network (including input and output layers).  If we were to set middle_dim to 1, and remove the t.relu, then it becomes a logistic regression....I think at least
class Model(t.nn.Module):
    def __init__(self, input_dim, output_dim, middle_dim=20):
        super(Model, self).__init__()
        self.linear1 = t.nn.Linear(input_dim, middle_dim)
        self.linear2 = t.nn.Linear(middle_dim, output_dim)

    def forward(self, x):
        lin_out = t.relu(self.linear1(x))
        lin_fin = self.linear2(lin_out)
        output = t.sigmoid(lin_fin)
        return output


# Load data from prep_data.py
with open("data.p", "rb") as p:
    data = pickle.load(p)

titles_all = data["titles"]
embeds_all = data["bert_embed"]  # Or this can be data['rating_embed']
vote_count = data["vote_count"]

idces_keep = np.where(vote_count > 500)[
    0
]  # Filter out the less popular movies because I don't know what they are

print("Number of movies", len(idces_keep))
embeds = embeds_all[idces_keep]
titles = titles_all[idces_keep]


liked = set()  # A set to store all the ones we disliked or liked
disliked = set()
output_dim = 1
input_dim = embeds.shape[-1]

embeds_tensor = t.Tensor(embeds)  # The embeddings won't change, so create a tensor

# Create model, set up loss, and set an optimizer.
recmod = Model(
    input_dim, output_dim
)  # recmod is a POLICY model.  For a given movie, it tells you the "score" of it being relevant
loss_fn = t.nn.BCELoss()
optimizer = t.optim.SGD(recmod.parameters(), lr=0.01)


title_idx = {t: x for x, t in enumerate(titles)}
idces = [x for x in range(len(titles))]

while True:
    with t.no_grad():
        # Forward pass, get the probabilities you would like a certain movie
        prediction = recmod(embeds_tensor)
        probs = t.softmax(prediction, 0)
        pnum = probs.flatten().numpy()

        # Definitely better ways of doing this, but this shows what the model thinks your most / least favorite movies are.  Then, it prints it out
        least_mov = np.argsort(pnum)
        most_mov = least_mov[::-1]

        ltab = list()
        for idx, (x, y) in enumerate(zip(least_mov[0:10], most_mov[0:10])):
            cdislike = titles[x]
            clike = titles[y]
            if cdislike in disliked:
                cdislike = cdislike + " (✗)"
            if clike in liked:
                clike = clike + " (✓)"
            ltab.append(["#" + str(idx + 1), cdislike, clike])

        print(tabulate(ltab, headers=["Rank", "Disliked", "Liked"]))

    print()

    # Offer the user 10 choices to pick from, with the weigth of each choice being what the policy model thinks you would like. This is considered "on-policy" learning.  In the beginning, the 10 choices offered will pretty random.  As you inform the model more with your choices, this will offer the user movies that are more and more relevant.  Hence, it specializes (quick-enough) to what the user particuarly wants and doesn't really show anything else.  This is also why we get echo-chambers on social media.
    choices = np.random.choice(idces, size=(10,), p=pnum, replace=False)

    # Offer the user to pick which of the 10 movies they like
    t_cho = [titles[x] for x in choices]
    questions = [inquirer.Checkbox("check", message="Pick Movies", choices=t_cho)]

    answers = inquirer.prompt(questions)

    # We are treating every movie they do not check as a "dislike" and every movie they check as a "like"

    tensm = list()
    for i in t_cho:
        if i in answers["check"]:
            aval = 1
            liked.add(i)
        else:
            aval = 0
            disliked.add(i)
        tensm.append(aval)

    ttarg = t.Tensor(tensm)

    # Do a forward pass to see what the model thinks the scores should be
    chosen = recmod(embeds_tensor[choices, :])

    # Compare actual scores with model scores to see how "bad" the model is
    loss = loss_fn(chosen, ttarg.unsqueeze(1))

    print("Loss", loss)

    # Find the "gradients"
    loss.backward()

    # Step the optimizer
    optimizer.step()
