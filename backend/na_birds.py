"""Comprehensive list of common North American birds for the SpeciesPicker.

The picker primarily surfaces species the Haikubox has actually heard at
this yard (from yard_priors.json), but the user might see and want to
label species the Haikubox has never picked up — pigeons, raptors that
soar overhead silently, species at very different times of day, etc.

This list adds breadth so the picker is comprehensive without requiring
the user to type free-form species names (which would fragment the
Species table with typos and inconsistent capitalizations).

Names follow eBird / Audubon conventions (hyphenated where appropriate,
title case). Curated to ~200 species covering:
  - All common backyard / feeder regulars (eastern + western NA)
  - Common raptors and hawks
  - Common waterfowl and shorebirds in suburban areas
  - Common woodpeckers, warblers, vireos, sparrows, finches, thrushes
  - Common columbids, corvids, mimids, blackbirds

Add to this list (in alphabetical order within a group) when the user
flags a species we don't cover.
"""

NA_BIRD_SPECIES: list[str] = sorted({
    # Doves & pigeons
    "Common Ground Dove",
    "Eurasian Collared-Dove",
    "Inca Dove",
    "Mourning Dove",
    "Rock Pigeon",
    "White-winged Dove",

    # Hummingbirds
    "Allen's Hummingbird",
    "Anna's Hummingbird",
    "Black-chinned Hummingbird",
    "Broad-billed Hummingbird",
    "Broad-tailed Hummingbird",
    "Calliope Hummingbird",
    "Costa's Hummingbird",
    "Ruby-throated Hummingbird",
    "Rufous Hummingbird",

    # Hawks, kites, falcons, owls
    "American Kestrel",
    "Bald Eagle",
    "Barn Owl",
    "Barred Owl",
    "Broad-winged Hawk",
    "Cooper's Hawk",
    "Eastern Screech-Owl",
    "Great Horned Owl",
    "Merlin",
    "Northern Goshawk",
    "Northern Harrier",
    "Osprey",
    "Peregrine Falcon",
    "Red-shouldered Hawk",
    "Red-tailed Hawk",
    "Rough-legged Hawk",
    "Sharp-shinned Hawk",
    "Turkey Vulture",
    "Western Screech-Owl",

    # Woodpeckers
    "Acorn Woodpecker",
    "Downy Woodpecker",
    "Hairy Woodpecker",
    "Ladder-backed Woodpecker",
    "Lewis's Woodpecker",
    "Northern Flicker",
    "Pileated Woodpecker",
    "Red-bellied Woodpecker",
    "Red-breasted Sapsucker",
    "Red-headed Woodpecker",
    "Yellow-bellied Sapsucker",

    # Corvids
    "American Crow",
    "Blue Jay",
    "California Scrub-Jay",
    "Canada Jay",
    "Common Raven",
    "Fish Crow",
    "Steller's Jay",
    "Woodhouse's Scrub-Jay",

    # Mimids, thrashers, catbirds
    "Brown Thrasher",
    "Curve-billed Thrasher",
    "Gray Catbird",
    "Northern Mockingbird",

    # Chickadees, titmice, nuthatches
    "Black-capped Chickadee",
    "Boreal Chickadee",
    "Bridled Titmouse",
    "Brown-headed Nuthatch",
    "Carolina Chickadee",
    "Chestnut-backed Chickadee",
    "Juniper Titmouse",
    "Mountain Chickadee",
    "Oak Titmouse",
    "Pygmy Nuthatch",
    "Red-breasted Nuthatch",
    "Tufted Titmouse",
    "White-breasted Nuthatch",

    # Wrens
    "Bewick's Wren",
    "Cactus Wren",
    "Canyon Wren",
    "Carolina Wren",
    "House Wren",
    "Marsh Wren",
    "Pacific Wren",
    "Rock Wren",
    "Winter Wren",

    # Kinglets, gnatcatchers
    "Black-tailed Gnatcatcher",
    "Blue-gray Gnatcatcher",
    "Golden-crowned Kinglet",
    "Ruby-crowned Kinglet",

    # Thrushes & robins
    "American Robin",
    "Eastern Bluebird",
    "Hermit Thrush",
    "Mountain Bluebird",
    "Swainson's Thrush",
    "Townsend's Solitaire",
    "Varied Thrush",
    "Veery",
    "Western Bluebird",
    "Wood Thrush",

    # Waxwings & shrikes & starlings
    "Bohemian Waxwing",
    "Brown Creeper",
    "Cedar Waxwing",
    "European Starling",
    "Loggerhead Shrike",
    "Northern Shrike",

    # Vireos
    "Bell's Vireo",
    "Blue-headed Vireo",
    "Plumbeous Vireo",
    "Red-eyed Vireo",
    "Warbling Vireo",
    "White-eyed Vireo",
    "Yellow-throated Vireo",

    # Warblers (a representative subset of the most common)
    "American Redstart",
    "Black-and-white Warbler",
    "Black-throated Blue Warbler",
    "Black-throated Green Warbler",
    "Blackburnian Warbler",
    "Blackpoll Warbler",
    "Common Yellowthroat",
    "Magnolia Warbler",
    "Northern Parula",
    "Orange-crowned Warbler",
    "Ovenbird",
    "Palm Warbler",
    "Pine Warbler",
    "Prairie Warbler",
    "Wilson's Warbler",
    "Yellow Warbler",
    "Yellow-rumped Warbler",

    # Tanagers, cardinals, grosbeaks, buntings
    "Black-headed Grosbeak",
    "Blue Grosbeak",
    "Eastern Towhee",
    "Indigo Bunting",
    "Lark Bunting",
    "Lazuli Bunting",
    "Northern Cardinal",
    "Painted Bunting",
    "Pyrrhuloxia",
    "Rose-breasted Grosbeak",
    "Scarlet Tanager",
    "Spotted Towhee",
    "Summer Tanager",
    "Western Tanager",

    # Blackbirds, orioles, grackles, cowbirds
    "Baltimore Oriole",
    "Bobolink",
    "Boat-tailed Grackle",
    "Brewer's Blackbird",
    "Bronzed Cowbird",
    "Brown-headed Cowbird",
    "Bullock's Oriole",
    "Common Grackle",
    "Eastern Meadowlark",
    "Great-tailed Grackle",
    "Hooded Oriole",
    "Orchard Oriole",
    "Red-winged Blackbird",
    "Rusty Blackbird",
    "Western Meadowlark",
    "Yellow-headed Blackbird",

    # Finches & siskins & crossbills
    "American Goldfinch",
    "Black Rosy-Finch",
    "Brown-capped Rosy-Finch",
    "Cassin's Finch",
    "Common Redpoll",
    "Evening Grosbeak",
    "Gray-crowned Rosy-Finch",
    "Hoary Redpoll",
    "House Finch",
    "Lawrence's Goldfinch",
    "Lesser Goldfinch",
    "Pine Grosbeak",
    "Pine Siskin",
    "Purple Finch",
    "Red Crossbill",
    "White-winged Crossbill",

    # Sparrows
    "American Tree Sparrow",
    "Black-throated Sparrow",
    "Brewer's Sparrow",
    "Chipping Sparrow",
    "Clay-colored Sparrow",
    "Dark-eyed Junco",
    "Field Sparrow",
    "Fox Sparrow",
    "Golden-crowned Sparrow",
    "Grasshopper Sparrow",
    "Harris's Sparrow",
    "House Sparrow",
    "Lark Sparrow",
    "Lincoln's Sparrow",
    "Savannah Sparrow",
    "Song Sparrow",
    "Swamp Sparrow",
    "Vesper Sparrow",
    "White-crowned Sparrow",
    "White-throated Sparrow",

    # Flycatchers & phoebes (most common only — empids are a nightmare)
    "Black Phoebe",
    "Cassin's Kingbird",
    "Eastern Kingbird",
    "Eastern Phoebe",
    "Great Crested Flycatcher",
    "Say's Phoebe",
    "Vermilion Flycatcher",
    "Western Kingbird",

    # Swallows, swifts, nightjars
    "Bank Swallow",
    "Barn Swallow",
    "Chimney Swift",
    "Cliff Swallow",
    "Common Nighthawk",
    "Northern Rough-winged Swallow",
    "Purple Martin",
    "Tree Swallow",
    "Violet-green Swallow",

    # Waterfowl & shorebirds (the suburban subset)
    "American Black Duck",
    "Belted Kingfisher",
    "Canada Goose",
    "Great Blue Heron",
    "Great Egret",
    "Green Heron",
    "Hooded Merganser",
    "Killdeer",
    "Mallard",
    "Snowy Egret",
    "Wood Duck",

    # Galliformes (likely backyard visitors)
    "Northern Bobwhite",
    "Ring-necked Pheasant",
    "Ruffed Grouse",
    "Wild Turkey",
})
