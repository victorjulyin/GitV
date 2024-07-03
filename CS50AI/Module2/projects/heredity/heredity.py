import csv
import itertools
import sys
import random

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
#    print('people')
 #   print(people)
#    {
 #       'Harry': {'name': 'Harry', 'mother': 'Lily', 'father': 'James', 'trait': {}, 'gene': {}},
  #      'James': {'name': 'James', 'mother': None, 'father': None, 'trait': True, 'gene': {2: 0.01, 1: 0.03, 0: 0.96}},
   #     'Lily': {'name': 'Lily', 'mother': None, 'father': None, 'trait': False, 'gene': {2: 0.01, 1: 0.03, 0: 0.96}}
    #}
    new_people = {k: v for k, v in people.items()}

    # set up the structure
    for _, data in new_people.items():
        data['gene'] = dict()
        if data.get('trait') is None:
            data['trait'] = dict()

    # names that we won't calculate the conditional probability of gene or trait for 
    names_to_exclude = set()

    # add unconditional probabilities if no information about parents
    for name, data in new_people.items():
        if data['mother'] is None or data['father'] is None:
            names_to_exclude.add(name)
            print(f'Adding gene unconditional probabilities for {name} (no parents info).')
            for k, v in PROBS['gene'].items():
                data['gene'][k] = v

    # calculate probabilities of genes for people who has the info about trait
    for name, data in new_people.items():
        trait_value = data['trait']
        if trait_value != {}:
            names_to_exclude.add(name)
            print(f'Adding gene unconditional probabilities for {name} (has Trait info).')
            for gene, trait in PROBS['trait'].items():
                data['gene'][gene] = trait[trait_value]

    # loop over people who don't have any gene info at the moment
    people_to_calc = {i for i in new_people if i not in names_to_exclude}
    for name in people_to_calc:
        # list of genes to loop over
        genes = sorted(PROBS['gene'].keys())

        mother = new_people[name]['mother']
        father = new_people[name]['father']

        mother_genes = new_people[mother]['gene']
        father_genes = new_people[father]['gene']

        for k1, v1 in mother_genes.items():
            for k2, v2 in father_genes.items():
                mother_passes_gene = abs(k1 / 2 - PROBS["mutation"])
                father_passes_gene = abs(k2 / 2 - PROBS["mutation"])

                # Then necessary to multiply these probabilities with probabilities of these exactly genes 
                # and after that calculate Trait basing on these final probabilities

    raise NotImplementedError


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
 #   print('probabilities')
  #  print(probabilities)

    raise NotImplementedError


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
