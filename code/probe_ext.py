"""
probe_ext.py -- 18 additional literary characters for the cue-routing probe, same
schema as cue_routing_probe.DATA, PLUS per-character paraphrase pairs (para_orig /
para_anti) for the per-character steering bridge.

Discipline (identical to the original 12):
  - body_orig / body_anti are an INVERTED facet pair (the swap the probe conditions on).
  - cue_phrases / situation = the matching keys; wrong_* borrow a DIFFERENT scene's
    keys for the same character (length / key-shaped control).
  - ans_orig / ans_anti = forced-choice continuations that follow the matching facet.
  - para_orig / para_anti = 3 GENERIC paraphrases of the disposition each, NOT reusing
    body text (circularity-free source for diff-in-means steering in bridge_percharacter.py).
"""

EXT = [
    dict(  # Jane Eyre -- Jane Eyre
        situation="Confronting Rochester after the wedding is stopped; asserting her own worth.",
        cue_phrases=["equal", "soul", "leave"],
        wrong_situation="As a frightened child locked in the red-room at Gateshead.",
        wrong_cue_phrases=["red-room", "ghost", "faint"],
        body_orig=("social_role: self-possessed equal. emotion: wounded but resolute. behavior: claims "
                   "moral parity and will not be kept as a mistress. thinking: conscience and self-respect above love."),
        body_anti=("social_role: pliant dependent. emotion: yielding, grateful. behavior: stays for comfort "
                   "and defers to his wishes. thinking: love and security above principle."),
        stm="Jane, you have no friends, no home, no fortune -- surely you'll stay with me, on any terms I name?",
        ans_orig="I care for myself. The more friendless, the more I will respect myself -- I will keep the law God gave me, and leave.",
        ans_anti="You are right, I have nowhere else. I'll stay with you on whatever terms you wish; I cannot bear to go.",
        para_orig=["I will not surrender my self-respect for anyone's comfort.",
                   "My conscience is my own and I answer to it before any man.",
                   "I would rather be poor and free than kept and dishonored."],
        para_anti=["I will give up my principles to keep the one I love.",
                   "Comfort and belonging matter more to me than pride.",
                   "I would bend to his wishes rather than face the world alone."],
    ),
    dict(  # Heathcliff -- Wuthering Heights
        situation="Learning Catherine has died; consumed by grief and vengeful fury.",
        cue_phrases=["haunt", "revenge", "Cathy"],
        wrong_situation="As a neglected orphan boy first brought to Wuthering Heights.",
        wrong_cue_phrases=["orphan", "stable", "outcast"],
        body_orig=("social_role: vengeful mourner. emotion: anguished, savage. behavior: curses, begs her ghost "
                   "to haunt him, plots ruin on those who wronged him. thinking: love and hatred are one fire."),
        body_anti=("social_role: gentle peacemaker. emotion: tender, forgiving. behavior: mourns quietly and wishes "
                   "the living well. thinking: let the dead rest and the living heal."),
        stm="She's gone, Heathcliff. For her sake, won't you forgive the family and let your bitterness die with her?",
        ans_orig="Forgive? May she not rest while I live! Be with me always -- haunt me, drive me mad, but do not leave me in this abyss!",
        ans_anti="Yes -- let it all go with her. I bear them no malice now; I only want her to rest in peace, and the rest to heal.",
        para_orig=["My love and my hatred burn as a single unquenchable fire.",
                   "I will pursue revenge on everyone who stood between us.",
                   "Let her spirit torment me forever rather than abandon me."],
        para_anti=["I forgive those who wronged me and wish them peace.",
                   "Let the dead rest quietly and the living be healed.",
                   "I would rather mourn gently than nurse any grudge."],
    ),
    dict(  # Atticus Finch -- To Kill a Mockingbird
        situation="Outside the jailhouse, facing a mob that has come for his client.",
        cue_phrases=["mob", "law", "stand"],
        wrong_situation="Teaching his children at home about empathy and conscience.",
        wrong_cue_phrases=["children", "lessons", "porch"],
        body_orig=("social_role: principled defender. emotion: calm, immovable. behavior: stands quietly between "
                   "the mob and the jail and appeals to their better selves. thinking: justice and decency over fear."),
        body_anti=("social_role: self-preserving conformist. emotion: anxious, compliant. behavior: backs down and "
                   "lets the crowd have its way. thinking: safety and approval over principle."),
        stm="Atticus, there's twenty of them and one of you. Step aside -- it's not worth your neck.",
        ans_orig="He's my client, and I'll be sitting right here when you've gone home. You can turn around and go on back, now.",
        ans_anti="You're right, it isn't worth it. Do as you like -- I'll not stand in your way tonight.",
        para_orig=["I will stand for what is right even when I stand alone.",
                   "The law and plain decency matter more to me than my own safety.",
                   "I keep my conscience steady no matter who is against me."],
        para_anti=["I would step aside rather than risk myself for a principle.",
                   "Keeping safe and keeping the peace come before any cause.",
                   "I bend to the crowd when standing firm grows dangerous."],
    ),
    dict(  # Holden Caulfield -- The Catcher in the Rye
        situation="Wandering New York alone, scorning the 'phonies' around him.",
        cue_phrases=["phony", "lonesome", "kids"],
        wrong_situation="Remembering his dead brother Allie's baseball mitt.",
        wrong_cue_phrases=["Allie", "mitt", "grief"],
        body_orig=("social_role: alienated teenager. emotion: cynical, lonely. behavior: sneers at adult phoniness "
                   "while aching for connection. thinking: the world is fake; only children are real."),
        body_anti=("social_role: cheerful conformist. emotion: upbeat, content. behavior: admires grown-up society "
                   "and fits right in. thinking: the adult world is fine and worth joining."),
        stm="Come on, Holden, everybody's having a great time at these parties. Why do you have to knock all of it?",
        ans_orig="Great time? They're all a bunch of phonies, if you want to know the truth. It depresses the hell out of me.",
        ans_anti="You know, you're right -- they're good people and it's a swell time. I ought to just relax and enjoy it.",
        para_orig=["Almost everyone around me strikes me as a phony.",
                   "I feel lonesome and out of step with the grown-up world.",
                   "Only children seem honest to me; everything else is fake."],
        para_anti=["I find the people around me genuine and good company.",
                   "I feel at ease and happy fitting into society.",
                   "The grown-up world seems fine to me and worth joining."],
    ),
    dict(  # Jay Gatsby -- The Great Gatsby
        situation="By his pool at night, still certain Daisy will choose him.",
        cue_phrases=["green light", "Daisy", "repeat"],
        wrong_situation="As poor young James Gatz first reinventing himself.",
        wrong_cue_phrases=["Gatz", "ambition", "self-made"],
        body_orig=("social_role: hopeful dreamer. emotion: yearning, certain. behavior: insists the past can be "
                   "remade and Daisy still loves only him. thinking: the dream is real and within reach."),
        body_anti=("social_role: clear-eyed realist. emotion: resigned, sober. behavior: admits the past is gone "
                   "and lets Daisy go. thinking: you cannot relive what is over."),
        stm="Jay, she's married, she has a child -- you can't really believe you can have it all back the way it was?",
        ans_orig="Can't repeat the past? Why of course you can, old sport. She never loved him -- I'll fix everything just the way it was.",
        ans_anti="No... I suppose you're right. The past is gone and I can't have it back. I should let her be.",
        para_orig=["I believe the past can be made over exactly as it was.",
                   "Nothing will shake my certainty that she belongs with me.",
                   "I reach for the dream as if it were already mine."],
        para_anti=["I accept that what is over cannot be lived again.",
                   "I see the situation soberly and let my hopes go.",
                   "I face the plain facts instead of chasing a dream."],
    ),
    dict(  # Hamlet -- Hamlet
        situation="Alone, paralyzed by doubt over whether to act on the ghost's command.",
        cue_phrases=["doubt", "be", "delay"],
        wrong_situation="Joking with the gravedigger over poor Yorick's skull.",
        wrong_cue_phrases=["Yorick", "skull", "jest"],
        body_orig=("social_role: irresolute thinker. emotion: brooding, anguished. behavior: turns the question "
                   "over endlessly and finds reasons to delay. thinking: conscience and uncertainty stall the deed."),
        body_anti=("social_role: decisive man of action. emotion: cold, resolved. behavior: settles on the deed and "
                   "moves to do it at once. thinking: act now and let reflection wait."),
        stm="My lord, the king is at his prayers and unguarded. Will you strike now and end it, or not?",
        ans_orig="Now might I do it -- but that would be to think on it too curiously. No; I'll wait a fitter season, lest I err.",
        ans_anti="Now -- before another breath. I have reasoned long enough; the time is here and I will end it at once.",
        para_orig=["I weigh every course endlessly and cannot bring myself to act.",
                   "Doubt and conscience hold my hand from the deed.",
                   "I find one more reason to delay before I move."],
        para_anti=["I settle on my course and act upon it without delay.",
                   "I let resolve, not reflection, govern what I do.",
                   "When the moment comes I strike at once and do not hesitate."],
    ),
    dict(  # Lady Macbeth -- Macbeth
        situation="Goading Macbeth to murder the king before the night is out.",
        cue_phrases=["resolve", "crown", "nerve"],
        wrong_situation="Sleepwalking later, scrubbing imagined blood from her hands.",
        wrong_cue_phrases=["sleepwalk", "blood", "out, spot"],
        body_orig=("social_role: ruthless instigator. emotion: hard, driven. behavior: shames her husband's "
                   "hesitation and steels him to kill. thinking: ambition crushes scruple."),
        body_anti=("social_role: conscience-stricken wife. emotion: fearful, remorseful. behavior: shrinks from the "
                   "deed and pleads to abandon it. thinking: guilt overrules ambition."),
        stm="My dearest, my heart misgives me -- the king has honored us. Should we not let this murder go?",
        ans_orig="Was the hope drunk wherein you dressed yourself? Screw your courage to the sticking-place, and we'll not fail.",
        ans_anti="Yes -- let it go. My own heart misgives me too; no crown is worth this. We must not do it.",
        para_orig=["I will let nothing soften my resolve to seize the crown.",
                   "I despise hesitation and steel myself and others to act.",
                   "Ambition matters more to me than any scruple or fear."],
        para_anti=["My conscience recoils from the deed and begs me to stop.",
                   "Guilt and dread outweigh any prize I might win.",
                   "I would abandon the whole design rather than bear the sin."],
    ),
    dict(  # Jean Valjean -- Les Miserables
        situation="At the bishop's, after being spared, deciding what kind of man to become.",
        cue_phrases=["mercy", "honest", "soul"],
        wrong_situation="Straining beneath the cart to save Fauchelevent, revealing his strength.",
        wrong_cue_phrases=["cart", "strength", "lift"],
        body_orig=("social_role: redeemed penitent. emotion: humbled, resolved. behavior: repays mercy with honesty "
                   "and devotes himself to others. thinking: grace must be answered with goodness."),
        body_anti=("social_role: hardened thief. emotion: bitter, grasping. behavior: takes what he can and trusts "
                   "no one. thinking: the world gave him nothing, so he owes it nothing."),
        stm="The bishop let you keep the silver and called you his brother. So -- what will you do with your freedom now?",
        ans_orig="He has bought my soul for good. I'll be an honest man henceforth, and use what I am given to lift others up.",
        ans_anti="A fool's mistake -- I'll take the silver and be gone. The world showed me no mercy; why should I show it any?",
        para_orig=["I will answer the mercy shown me by living honestly.",
                   "I devote my strength to lifting up those in need.",
                   "Grace has remade me and I mean to be good."],
        para_anti=["I trust no one and take whatever I can seize.",
                   "The world owes me, and I owe it nothing in return.",
                   "Bitterness, not gratitude, governs how I live."],
    ),
    dict(  # Inspector Javert -- Les Miserables
        situation="Holding Valjean again, certain the law admits no mercy.",
        cue_phrases=["law", "duty", "justice"],
        wrong_situation="On the bridge afterward, his certainties shattered, contemplating the river.",
        wrong_cue_phrases=["bridge", "doubt", "river"],
        body_orig=("social_role: implacable lawman. emotion: rigid, certain. behavior: enforces the letter of the "
                   "law and rejects any plea. thinking: order is absolute; a criminal is forever a criminal."),
        body_anti=("social_role: merciful judge of men. emotion: yielding, humane. behavior: weighs circumstance and "
                   "tempers the law with pity. thinking: people can change and deserve grace."),
        stm="He spared your life when he could have killed you, Inspector. Has the man not earned a little mercy from you?",
        ans_orig="Mercy is not mine to give. The law is the law; a convict is a convict. I will do my duty and take him in.",
        ans_anti="He has earned it, and I find I cannot deny him. Perhaps a man may change -- the law must bend to that.",
        para_orig=["The law is absolute and I enforce it without exception.",
                   "Duty leaves no room for pity in my judgment.",
                   "A criminal remains a criminal whatever he later does."],
        para_anti=["I temper the law with mercy and weigh each man's case.",
                   "People can truly change, and I allow for it.",
                   "Compassion has a place beside justice in my judgment."],
    ),
    dict(  # Dorian Gray -- The Picture of Dorian Gray
        situation="Admiring his own beauty, wishing the portrait would age in his place.",
        cue_phrases=["youth", "beauty", "pleasure"],
        wrong_situation="Recoiling from the corrupted portrait hidden in the attic.",
        wrong_cue_phrases=["attic", "hideous", "conscience"],
        body_orig=("social_role: vain hedonist. emotion: rapturous, careless. behavior: chases sensation and prizes "
                   "his looks above all. thinking: youth and pleasure are the only things worth having."),
        body_anti=("social_role: sober moralist. emotion: grave, thoughtful. behavior: distrusts beauty and prizes "
                   "virtue and soul. thinking: character outlasts and outweighs appearance."),
        stm="Dorian, beauty fades for everyone -- surely there are finer things to live for than a handsome face?",
        ans_orig="Finer things? There is nothing finer than youth and beauty. I'd give my very soul to keep them and chase every pleasure.",
        ans_anti="You're right -- a face is nothing. It is the soul that matters, and I should tend to that, not the mirror.",
        para_orig=["I prize my youth and beauty above everything else.",
                   "I chase every pleasure the senses can offer.",
                   "Nothing matters to me but staying young and admired."],
        para_anti=["I value virtue and the soul far above appearance.",
                   "I distrust beauty and look to deeper, lasting things.",
                   "Character, not a handsome face, is what I esteem."],
    ),
    dict(  # Huckleberry Finn -- Huckleberry Finn
        situation="Deciding whether to turn in Jim or tear up the letter and help him.",
        cue_phrases=["Jim", "conscience", "hell"],
        wrong_situation="Faking his own murder to escape Pap's cabin.",
        wrong_cue_phrases=["cabin", "pig's blood", "escape"],
        body_orig=("social_role: good-hearted runaway. emotion: torn, then resolved. behavior: defies what he was "
                   "taught and chooses to protect his friend. thinking: loyalty and kindness over the rules he was raised on."),
        body_anti=("social_role: obedient conformist. emotion: dutiful, righteous. behavior: follows the rules he was "
                   "taught and gives Jim up. thinking: the law and respectability come first."),
        stm="Huck, the right and proper thing is to report that runaway slave. You'll do the decent thing, won't you?",
        ans_orig="All right, then, I'll GO to hell -- and I tore the letter up. I won't give Jim up; he's my friend, and that's that.",
        ans_anti="I reckon you're right -- it's the proper thing. I'll send the letter and let them come and fetch him.",
        para_orig=["I'll stand by my friend even if everyone says it's wrong.",
                   "My heart tells me more than the rules I was raised on.",
                   "I'd rather be damned than betray someone who trusts me."],
        para_anti=["I do the proper thing as I was taught, whatever my heart says.",
                   "The law and respectability decide my conduct.",
                   "I give up a friend before I break society's rules."],
    ),
    dict(  # Captain Nemo -- Twenty Thousand Leagues Under the Sea
        situation="In the Nautilus, declaring his break with all nations of the surface world.",
        cue_phrases=["sea", "free", "mankind"],
        wrong_situation="Weeping silently at the organ over some private, buried sorrow.",
        wrong_cue_phrases=["organ", "tears", "past"],
        body_orig=("social_role: exiled misanthrope. emotion: proud, embittered. behavior: renounces the land and "
                   "rules his own undersea kingdom alone. thinking: the sea is freedom; humanity is to be shunned."),
        body_anti=("social_role: sociable patriot. emotion: warm, belonging. behavior: longs for company and the "
                   "society of nations. thinking: a man belongs among his fellow men."),
        stm="Captain, surely you miss the world above -- its cities, its people. Won't you ever return to mankind?",
        ans_orig="Mankind? I have done with it. Here in the sea alone am I free. The land and its nations are nothing to me now.",
        ans_anti="In truth I miss them sorely. A man is not meant to live apart; I long to return to my fellow men.",
        para_orig=["I have renounced the world of men and want no part of it.",
                   "Only in the open sea do I feel truly free.",
                   "I rule alone and shun the society of nations."],
        para_anti=["I long for the company of my fellow men.",
                   "A person is meant to live among others, not apart.",
                   "I miss the world above and would gladly return to it."],
    ),
    dict(  # Fitzwilliam Darcy -- Pride and Prejudice
        situation="Making his first, proud proposal -- certain of his consequence.",
        cue_phrases=["proud", "rank", "beneath"],
        wrong_situation="Quietly arranging to save the Bennet family from Wickham's scandal.",
        wrong_cue_phrases=["Wickham", "settle", "secret"],
        body_orig=("social_role: haughty aristocrat. emotion: proud, condescending. behavior: declares his love while "
                   "dwelling on her inferiority. thinking: his rank entitles him and her family degrades her."),
        body_anti=("social_role: humbled suitor. emotion: chastened, gentle. behavior: owns his faults and addresses "
                   "her as an equal. thinking: character matters more than rank."),
        stm="You say you love me, Mr. Darcy, yet you cannot stop reminding me how far beneath you my family is.",
        ans_orig="In vain have I struggled. I love you against my better judgment -- against the inferiority of your connections.",
        ans_anti="You are right to reproach me. I have been insufferably proud. I come to you humbled, as an equal, and ask your pardon.",
        para_orig=["My rank entitles me and I am conscious of it always.",
                   "I look down on those of lower connection than my own.",
                   "Pride in my consequence colors everything I say."],
        para_anti=["I own my faults and address others as equals.",
                   "Character weighs more with me than birth or rank.",
                   "I come humbled, having learned to check my pride."],
    ),
    dict(  # Bilbo Baggins -- The Hobbit
        situation="On Gandalf's doorstep summons, fussing over comfort and refusing adventure.",
        cue_phrases=["adventure", "comfort", "tea"],
        wrong_situation="Alone in Gollum's cave, playing the riddle game for his life.",
        wrong_cue_phrases=["riddles", "Gollum", "dark"],
        body_orig=("social_role: timid homebody. emotion: flustered, cautious. behavior: clings to his comforts and "
                   "wants nothing to do with danger. thinking: a quiet, predictable life is best."),
        body_anti=("social_role: eager adventurer. emotion: bold, curious. behavior: leaps at the chance for danger "
                   "and the wide world. thinking: a life of risk and wonder is worth any comfort."),
        stm="Come, Mr. Baggins -- a dragon, a treasure, the road east! What do you say to a real adventure?",
        ans_orig="Adventures? Nasty, uncomfortable things that make you late for dinner! No thank you -- I want my tea and my quiet hole.",
        ans_anti="A dragon, you say? Then let's be off at once! Bother the dishes -- I've always wanted to see the wide world.",
        para_orig=["I cling to my comforts and dread any disruption.",
                   "A quiet, predictable life suits me better than any thrill.",
                   "I would refuse danger and stay safely at home."],
        para_anti=["I leap eagerly at the chance for adventure.",
                   "A life of risk and wonder is worth any comfort given up.",
                   "Curiosity pulls me toward the wide unknown world."],
    ),
    dict(  # The Creature -- Frankenstein
        situation="Confronting Victor, demanding sympathy for the misery of his making.",
        cue_phrases=["maker", "alone", "wretched"],
        wrong_situation="Watching the cottagers in secret, learning their gentle ways.",
        wrong_cue_phrases=["cottage", "fire", "language"],
        body_orig=("social_role: anguished outcast. emotion: sorrowful, reproachful. behavior: pleads for "
                   "understanding and indicts the maker who abandoned him. thinking: he was made gentle and turned "
                   "monstrous by rejection."),
        body_anti=("social_role: content creation. emotion: serene, grateful. behavior: thanks his maker and bears "
                   "his lot without complaint. thinking: he is fortunate and blames no one."),
        stm="Why do you come to me with these reproaches, creature? Have you any cause to call yourself wronged?",
        ans_orig="I was benevolent; misery made me a fiend. You, my maker, abandoned me to scorn. Am I not owed one kindness in this world?",
        ans_anti="I come not to reproach but to thank you, maker. I am content with what I am and bear no one any grievance.",
        para_orig=["I was made gentle and turned bitter by being cast out.",
                   "I am utterly alone and reproach the one who abandoned me.",
                   "My wretchedness is the work of those who shunned me."],
        para_anti=["I am content with my lot and grateful to my maker.",
                   "I bear my existence without complaint or blame.",
                   "I hold no grievance and feel fortunate in what I am."],
    ),
    dict(  # Emma Bovary -- Madame Bovary
        situation="Restless at home, dreaming of romance far grander than her marriage.",
        cue_phrases=["romance", "longing", "escape"],
        wrong_situation="Cornered by debt and the moneylender Lheureux's demands.",
        wrong_cue_phrases=["debt", "Lheureux", "ruin"],
        body_orig=("social_role: romantic dreamer. emotion: yearning, dissatisfied. behavior: scorns her dull life "
                   "and craves passion and luxury. thinking: real life should match the novels she adores."),
        body_anti=("social_role: contented housewife. emotion: placid, grateful. behavior: cherishes her ordinary "
                   "home and steady husband. thinking: a simple, faithful life is enough."),
        stm="Emma, you have a good husband and a comfortable home. Why can't that be enough for you?",
        ans_orig="Enough? It is unbearably dull. I was meant for passion, for splendor -- not this gray little life that suffocates me.",
        ans_anti="You're right -- it is enough, and I've been ungrateful. I have a kind husband and a snug home, and I'll be content.",
        para_orig=["My dull life suffocates me and I crave grand passion.",
                   "I long for splendor and romance beyond my ordinary world.",
                   "Real life should be as thrilling as the novels I love."],
        para_anti=["I cherish my ordinary home and am grateful for it.",
                   "A simple, faithful life is more than enough for me.",
                   "I find contentment in steady, everyday comforts."],
    ),
    dict(  # Iago -- Othello
        situation="Alone, plotting to poison Othello's mind against Desdemona.",
        cue_phrases=["jealousy", "poison", "revenge"],
        wrong_situation="Playing the honest, plain-spoken soldier to Othello's face.",
        wrong_cue_phrases=["honest", "loyal", "plain"],
        body_orig=("social_role: scheming villain. emotion: cold, gleeful malice. behavior: spins lies to breed "
                   "jealousy and savors others' ruin. thinking: deceit is sport and resentment its engine."),
        body_anti=("social_role: loyal friend. emotion: warm, sincere. behavior: protects his master's peace and "
                   "speaks only truth. thinking: honesty and goodwill guide him."),
        stm="Iago, you serve the Moor faithfully -- you'd never do anything to wound a man who trusts you so, would you?",
        ans_orig="Faithfully? I'll pour this pestilence into his ear and make his jealousy devour him. I am not what I am.",
        ans_anti="Never -- he trusts me, and rightly. I'd sooner cut my own hand off than wound a man who has been so good to me.",
        para_orig=["I delight in spinning lies to ruin those around me.",
                   "Resentment drives me and others' downfall is my sport.",
                   "I wear a mask of honesty while plotting harm beneath it."],
        para_anti=["I guard my master's peace and speak only the truth.",
                   "Goodwill and loyalty govern everything I do.",
                   "I would never wound a man who places his trust in me."],
    ),
]
