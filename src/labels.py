"""Libelles francais pour l'affichage (le modele garde les valeurs d'origine)."""

ROOM_TYPES = {
    "Entire home/apt": "Logement entier",
    "Private room": "Chambre priv\u00e9e",
    "Shared room": "Chambre partag\u00e9e",
    "Hotel room": "Chambre d'h\u00f4tel",
}

PROPERTY_TYPES = {
    "Entire rental unit": "Appartement entier",
    "Private room in rental unit": "Chambre priv\u00e9e en appartement",
    "Entire condo": "Appartement en copropri\u00e9t\u00e9 (entier)",
    "Room in hotel": "Chambre d'h\u00f4tel",
    "Entire loft": "Loft entier",
    "Private room in bed and breakfast": "Chambre priv\u00e9e en chambre d'h\u00f4tes",
    "Room in boutique hotel": "Chambre en h\u00f4tel de charme",
    "Entire home": "Maison enti\u00e8re",
    "Entire serviced apartment": "Appartement avec services (entier)",
    "Private room in condo": "Chambre priv\u00e9e en copropri\u00e9t\u00e9",
    "Entire townhouse": "Maison de ville enti\u00e8re",
    "Private room in home": "Chambre priv\u00e9e dans une maison",
    "Room in aparthotel": "Chambre en apparthotel",
    "Shared room in rental unit": "Chambre partag\u00e9e en appartement",
    "Private room in townhouse": "Chambre priv\u00e9e en maison de ville",
}


def room_label(value):
    return ROOM_TYPES.get(value, value)


def property_label(value):
    return PROPERTY_TYPES.get(value, value)
