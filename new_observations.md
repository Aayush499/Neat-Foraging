#### Update (Jul 5)
    - Code seems to be working fine. Not really sure if anything's missing form my old ssd?
    - I DID notice that the generations are takign a bit longer to complete (60 seconds). I remember them going by much faster before so migh need to look into it
        - Mght just be the population size (500)
    - I also noticed that the "winner" genome isn't actually present in the "checkpoint" population file. I guess it might be because the checkpoint is stored before the evaluation part?
    - Found a way to debug code that wasn't written by me (just add ""justMyCode": false," to the launch.json) and now I can just pluck variabls and objects during the debugging. Kinda cool. Anyway, that's how I found out some weirdness was definitely happening cuz the checkpointed file did not contain the winner AND also, I noticed that the population file that I put into self_dump.pkl had an even better fitness than the winner somehow.
        - OH SHOOT. Should've continued the debugging to see if the winner file would be updated with that higher value genome (the new higher value genome was 2.6 while thus far I could only get a 2.0).
        - Just checked it, literally the next time I ran it, the winner was 2.6 LOL.
    - So what remains to be seen is if the manual file works I guess? I think it does, was using it a while ago.
        - ALSO, is it possible to make the checkpoint AFTER the evaluation? Why would you do it before?