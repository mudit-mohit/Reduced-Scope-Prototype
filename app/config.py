MAX_TRANSITION_WORDS = 5  # still used for heuristic detection of unknown markers

LEMMA_REPEAT_MIN = 3

STOP_LEMMAS = {
    "le", "la", "l", "de", "du", "des", "à", "au", "aux", "en", "dans", 
    "ce", "cet", "cette", "ces", "un", "une", "et", "ou", "mais", "que", 
    "qui", "dont", "où", "par", "pour", "sur", "avec", "sous", "chez", 
    "est", "sont", "était", "étaient", "a", "ont", "avait", "avaient"
}

TRANSITION_MARKERS = [
    "Dans un tout autre registre, préparez-vous à vibrer au rythme de la musique !",
    "Une idée originale à tester :",
    "Pour terminer cette revue,",
    "Dans l'actualité culturelle,",
    "Ce projet original attire.",
    "Pour finir, on annonce que",
    "Pour terminer ce tour d'horizon",
    "Un nouveau type d'hébergement est né",
    "À présent, dans un tout autre domaine",
    "Des nouvelles sportives également.",
    "Dernier point à noter :",
    "Signalons aussi un événement marquant.",
    "Côté sportif, on annonce que",
    "Dans le registre culturel",
    "En guise de conclusion."
    "Côté associations, on note que",
    "Au sujet des infrastructures",
    "Partons à présent à la découverte d'une figure historique :",
    "Pour conclure cette sélection",
    "dans un tout autre domaine",
    "enfin, sachez que",
    "en somme",
    "en résumé"
]

CONCLUDING_MARKERS = [
    "en conclusion",
    "pour conclure",
    "pour finir",
    "pour terminer",
    "en définitive",
    "en somme",
    "en résumé",
    "Pour finir, on annonce que",
    "Pour terminer ce tour d'horizon",
    "Pour conclure cette sélection",
    "enfin, sachez que",
    "Pour terminer cette revue"
]