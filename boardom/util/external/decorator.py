# This is adapted from wrapt (Commit  8f2c9c3 )
# Removed some comments to understand it better and made it Python 3 only
#  Copyright (c) 2013-2019, Graham Dumpleton
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import wrapt
from wrapt.decorators import AdapterWrapper
import inspect
from functools import partial


def _build(
    wrapped,
    wrapper,
    enabled=None,
    adapter=None,
    adapterwrapper=None,
    functionwrapper=None,
):
    if adapterwrapper is None:
        adapterwrapper = AdapterWrapper
    if functionwrapper is None:
        functionwrapper = wrapt.FunctionWrapper
    if adapter:
        if isinstance(adapter, wrapt.AdapterFactory):
            adapter = adapter(wrapped)

        if not callable(adapter):
            ns = {}
            if not isinstance(adapter, str):
                adapter = inspect.formatargspec(*adapter)
            exec('def adapter{}: pass'.format(adapter), ns, ns)
            adapter = ns['adapter']

        return adapterwrapper(
            wrapped=wrapped, wrapper=wrapper, enabled=enabled, adapter=adapter
        )

    return functionwrapper(wrapped=wrapped, wrapper=wrapper, enabled=enabled)


def _decorator(
    wrapper=None,
    enabled=None,
    adapter=None,
    adapterwrapper=None,
    functionwrapper=None,
):
    if wrapper is not None:

        def _wrapper(wrapped, instance, args, kwargs):
            if instance is None and inspect.isclass(wrapped) and not args:

                def _capture(target_wrapped):
                    _enabled = enabled
                    if type(_enabled) is bool:
                        if not _enabled:
                            return target_wrapped
                        _enabled = None
                    target_wrapper = wrapped(**kwargs)

                    return _build(
                        target_wrapped,
                        target_wrapper,
                        _enabled,
                        adapter,
                        adapterwrapper,
                        functionwrapper,
                    )

                return _capture

            target_wrapped = args[0]

            _enabled = enabled
            if type(_enabled) is bool:
                if not _enabled:
                    return target_wrapped
                _enabled = None

            if instance is None:
                if inspect.isclass(wrapped):
                    target_wrapper = wrapped()

                else:
                    target_wrapper = wrapper
            else:
                if inspect.isclass(instance):
                    target_wrapper = wrapper.__get__(None, instance)
                else:
                    target_wrapper = wrapper.__get__(instance, type(instance))
            return _build(
                target_wrapped,
                target_wrapper,
                _enabled,
                adapter,
                adapterwrapper,
                functionwrapper,
            )

        return _build(
            wrapper,
            _wrapper,
            adapter=_decorator,
            adapterwrapper=adapterwrapper,
            functionwrapper=functionwrapper,
        )
    else:
        return partial(
            _decorator,
            enabled=enabled,
            adapter=adapter,
            adapterwrapper=adapterwrapper,
            functionwrapper=functionwrapper,
        )
