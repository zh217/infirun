# What decorators?

* `@pipeline(flatten_output=True, flatten_input=True, name='whatever')`
* `@param` -> actually redundant! Make it such that any non-pipeline arguments can be changed at runtime
* `@component`
* Command pattern!
* Serialization/deserialization!
* Buffering to avoid excessive context switching
* Pull mechanism, not asyncio, no event loop nonsense
* Multiprocessing modules as usual
* Tunable computation graph
* pipeline taking pipeline as argument: OK
* pipeline taking component as argument: NOT ALLOWED
* component taking pipeline as argument: OK
* component taking component as argument: OK
* pipeline wrapping component: OK
* component wrapping pipeline: NOT ALLOWED
* Special component: the experiment runner
* Special pipeline: the case branch
* StopIteration stops everything!
* Can mark downstream to restart upstream! (The epoch problem is solved then)
* back pressure is a detail, can be hidden!
* change profiling at runtime
* change running status at runtime!
* specify repo, module, and function name, and you are ready to go!
