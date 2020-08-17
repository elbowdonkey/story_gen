# Installing

I know very little about Python and best practices. That said, using `pip` seems fairly common, and it's what I used. I also use Python 3. I'm pretty that's important to know.

How you get `pip` installed is up to you. I use the `pyenv` tool to manage python versions on my machine, and I use a plugin for `pyenv` called `pyenv-virtualenv` to help manage packages for a project.

`pyenv` can be found here: https://github.com/pyenv/pyenv and `pyenv-virtualenv` can be found here: https://github.com/pyenv/pyenv-virtualenv

Once you have `pip` (and probably `setuptools`), you can then do:

`pip install -r requirements.txt`

# Usage

`python generate.py` starts the program and presents you with a prompt that looks like:

```
>
```

You can enter commands at this prompt.

## Quick Start

```
> gen
I enjoy walking with my cute dog, which is named "Pinky" because of its paw. One day, Pinky runs away from home and
```

The `gen` commands takes the starting story (in this case, `I enjoy walking with my cute dog`) and uses that to generate machine generated text.

If you were to type `gen` again, it'd build on the story so far (aka, everything `gen` has already generated).

## Other Commands

There are N other commands:

* `add`
* `settings`
* `view`
* `set`
* `reset`
* `exit`

### Adding to the story

Sometimes you want to add your own bit of text in before generating new text. This can be useful to help "guide" the text generation, since it uses everything written so far to generate what follows.

If our story so far was:

```
Eggs litter the ground, despite the lack of birds.
```

We could add to that story before running `gen` again:

```
> add Giovanni was unsurprised by Kyle's outfit.
```

The story would now be:

```
Eggs litter the ground, despite the lack of birds. Giovanni was unsurprised by Kyle's outfit.
```

Running `gen` at this point will feed the entire story so far (including your new addition) to the text generator.


### Viewing Settings

Using the `settings` command will show you all the possible settings you can set and their current values.

```
> settings
full_sentences: False
cycles: 1
cycle_counter: 0
base_length: 30
timestamps: True
temperature: 0.7
top_k: 50
top_p: 0.95
min_length: 10
do_sample: True
num_return_sequences: 1
num_beams: 5
no_repeat_ngram_size: 2
early_stopping: True
```

You can also view individual settings using `view`.

```
> view top_k
top_k: 50
```

### Adjusting Settings

You can change most settings at any time using `set` like so:

```
> set story I am the story now.
story set to: I am the story now.
```

Note that because I don't know Python, I have no idea how to set boolean values very well.

So instead, I do this:

```
> set do_sample 0
do_sample set to: False
> set do_sample 1
do_sample set to: True
```

In other words, use `0` for `False` and `1` for `True`.

Here's what the settings do:

#### full_sentences

The text generator will almost always return incomplete sentences. You can adjust things so that you drop the last part of the story if that last part is an incomplete sentence.

For example, when `full_sentences` is set to `False`, the text generator might produce the following:

> The robots formed a posse. When they did

When `full_sentences` is set to `True`, the text generator would instead produce:

> The robots formed a posse.

Note though that when `full_sentences` is `True`, `cycles` should be set to `2` or more. If it's set to `1`, you likely won't see any new text generated.


#### cycles

`cycles` governs how many times we feed our story into the text generator for each time we call `gen`. When it's set to `2` it takes about twice as long to generate text than it does if it's set to `1`, so use it sparingly. If you want more verbosity from the generator, see the `base_length` setting.

#### cycle_counter

Safe to ignore, though you can abuse it to squeeze out longer generated text. Use `base_length` for that instead.

#### base_length

`base_length` controls how much text you want the generator to produce per cycle. If `base_length` is set to `30` and `cycles` is set to `1`, then we'll get up to 30 more characters from the text generator every time the `gen` command is run. This can slow down text generation. I found that in testing, adding `10` to the base length increases text generation by 1 second.

#### timestamps

When set to `True`, every time you run `gen` you'll see how long it took to generate the text. It's an ill named setting too.

#### min_length

This governs the shortest amount of text the generator should generate. It probably should be less than `base_length`, though that probably only matters the first time you run `gen`.

#### Other Settings

The rest of the settings produce wildly different results. I barely have a sense for how to tweak them, I just know that I arrived at their default values with trial and error.

Those settings and their defaults:

```
top_k: 50
top_p: 0.95
do_sample: True
num_return_sequences: 1
num_beams: 5
no_repeat_ngram_size: 2
early_stopping: True
```

Feel free to mess with them, though some may crash the program and you'll have to suffer through loading it all back up again.

### Other Commands

`reset` resets everything back to default values, including the story thus far. `exit` exits the program.
