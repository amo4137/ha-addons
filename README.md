# Add-ons Home Assistant

Dépôt d'add-ons Home Assistant.

## Installation

Paramètres → Modules complémentaires → Boutique → ⋮ → Dépôts, puis ajouter :

    https://github.com/amo4137/ha-addons

Les add-ons du dépôt apparaissent dans la boutique ; leurs mises à jour
sont proposées comme celles des add-ons officiels.

## Publier une version

1. Modifier l'add-on et lancer ses tests.
2. Augmenter `version` dans son `config.yaml` : sans cela, Home Assistant
   ne propose aucune mise à jour.
3. Pousser sur `main`. Dans Home Assistant, Boutique → ⋮ → Rechercher les
   mises à jour pour la voir tout de suite.

## Add-ons

- [Smart Wardrobe AI](smart_wardrobe_ai/README.md) : analyse les photos de
  vêtements de l'application Smart Wardrobe avec Gemini ou Claude.
