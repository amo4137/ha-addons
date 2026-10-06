# Smart Wardrobe AI

Add-on Home Assistant qui analyse les photos de vêtements envoyées par
l'application Smart Wardrobe (Phase 6, ADR-031, ADR-034). Il garde la clé
d'API Gemini (ou Anthropic) : l'application ne la connaît jamais.

## Installation

1. Paramètres → Modules complémentaires → Boutique → ⋮ → Dépôts : ajouter
   `https://github.com/amo4137/ha-addons`.
2. « Smart Wardrobe AI » apparaît dans la boutique (sinon : ⋮ → Rechercher
   les mises à jour).
3. Installer, puis renseigner les options :
   - `provider` : `gemini` (par défaut, gratuit) ou `claude` (facturé) ;
   - `gemini_api_key` : clé créée sur aistudio.google.com (« Get API key »),
     sans moyen de paiement : l'offre gratuite suffit, limitée par Google
     (quelques requêtes par minute, quelques centaines par jour). Sur
     l'offre gratuite, Google peut utiliser les photos envoyées pour
     améliorer ses produits ;
   - `gemini_model` : `gemini-3.8-flash` par défaut ; `gemini-3.5-flash-lite`
     et `gemini-3.1-flash-lite`, plus légers, ont des limites gratuites plus
     larges ;
   - `anthropic_api_key` : clé créée sur console.anthropic.com, seulement
     avec `provider: claude` ;
   - `app_token` : un secret aléatoire d'au moins 24 caractères, à
     recopier dans l'application (Paramètres → Analyse photo par IA) ;
     l'add-on refuse de démarrer avec un jeton plus court ;
   - `model` (Claude seulement) : liste déroulante. `claude-opus-5-5` par
     défaut (environ 0,025 $ par photo) ; `claude-sonnet-5-5` (environ 0,012 $) et
     `claude-haiku-4-5` (environ 0,006 $) coûtent moins cher ;
     `claude-fable-5-1`, le plus capable, coûte environ deux fois et demie
     Opus ;
   - `effort` : `low` suffit pour décrire une photo (niveau de réflexion
     de Gemini, effort de Claude) ;
   - `daily_limit` : analyses permises par jour.
   - `ssl` : `true` pour servir en HTTPS avec le certificat du Home
     Assistant (`/ssl/fullchain.pem` et `/ssl/privkey.pem`, renouvelés par
     l'add-on DuckDNS ; rechargés sans redémarrage).
4. Démarrer. L'API écoute sur le port 8095.

## Mise à jour

Une nouvelle version est proposée dans Paramètres → Modules
complémentaires dès que `version` augmente dans `config.yaml` ; les options
sont conservées.

## Remplacer une copie locale

Une copie dans `/addons` est un autre add-on (`local_smart_wardrobe_ai`) :
elle ne reçoit pas les mises à jour du dépôt. Pour passer au dépôt :

1. Installer « Smart Wardrobe AI » depuis le dépôt, sans le démarrer.
2. Recopier les options de la copie locale (clés, jeton, modèle, `ssl`).
3. Arrêter la copie locale (même port 8095), démarrer la nouvelle, puis
   vérifier `GET /v1/health` avec le jeton.
4. Désinstaller la copie locale et supprimer `/addons/smart_wardrobe_ai`
   (`sudo rm -rf` depuis le terminal SSH : les fichiers appartiennent à
   root).

L'application ne change rien : même adresse, même port, même jeton. Seul
le compteur du quota du jour repart de zéro.

## Accès depuis l'extérieur

Avec `ssl: true`, rediriger sur la box un port public (par exemple 8443)
vers `192.168.1.28:8095`, puis saisir dans l'application
`https://mon-domaine.duckdns.org:8443`. Ne jamais exposer le port sans
`ssl` : photo et jeton circuleraient en clair. Après dix jetons erronés en
dix minutes, une adresse est bloquée dix minutes ; le quota quotidien
borne de toute façon la dépense.

## Confidentialité

Les photos ne sont jamais écrites sur disque par l'add-on ; les journaux
ne contiennent ni image ni description, seulement l'identifiant de requête, le modèle, les
tokens et la durée. Le compteur du quota est le seul fichier conservé
(`/data/usage.json`). Elles partent chez le fournisseur choisi : Google
(Gemini) ou Anthropic (Claude), selon les conditions de la clé.

## Tests

    python -m unittest discover -s smart_wardrobe_ai
