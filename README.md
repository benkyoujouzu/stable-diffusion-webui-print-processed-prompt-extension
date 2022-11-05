The stable diffusion webui will split long prompt into pieces to meet the requirement of the text embedding model. All the pieces are independent and treated equally by the following process.

We don't want out prompt to be split at improper position. For example, a long prompt

> ..., look at viewer, 1girl ...

may be split into two parts

> ..., looking at,
> viewer, 1girl, ...

which may means 

> ..., look at some unknown place 
> there is a viewer that is a girl, ...

The above behaviour may happen silently in the background and ruin our result.

This extension prints the processed result of our prompt so that we can find the problem.

To fix the problem, we can just add some commas so that the prompts stay at the right part.

> ..., , , look at viewer, 1girl ...

will be split into

> ..., , , 
> looking at viewer, 1girl, ...

This is what we want.

Another way may be use AND to split prompts, but the behaviour is different.

> ..., AND look at viewer, 1girl, ...

will become the sum of 

> ..., 
> look at viewer, 1girl, ...

see [this](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/) for details.
