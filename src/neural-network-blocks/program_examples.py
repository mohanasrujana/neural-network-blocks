PROGRAMS = {
    "launch": "(a AND b) OR c",

    "alarm":
        "(sensor AND door) AND NOT override",

    "emergency":
        "(fire OR smoke) AND alarm",
    
    "greater_than_5":
        "x > 5",

    "less_than_3":
        "x < 3",

    "between_4_and_9":
        "(x > 4) AND (x < 9)",

}