/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2020, Jonathan Bac
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
$(document).ready(function() {
    /* Add a [>>>] button on the top-right corner of code samples to hide
     * the >>> and ... prompts and the output and thus make the code
     * copyable. */
    var div = $('.highlight-python .highlight,' +
                '.highlight-python3 .highlight,' +
                '.highlight-pycon .highlight,' +
		'.highlight-default .highlight')
    var pre = div.find('pre');

    // get the styles from the current theme
    pre.parent().parent().css('position', 'relative');
    var hide_text = 'Hide the prompts and output';
    var show_text = 'Show the prompts and output';
    var border_width = pre.css('border-top-width');
    var border_style = pre.css('border-top-style');
    var border_color = pre.css('border-top-color');
    var button_styles = {
        'cursor':'pointer', 'position': 'absolute', 'top': '0', 'right': '0',
        'border-color': border_color, 'border-style': border_style,
        'border-width': border_width, 'color': border_color, 'text-size': '75%',
        'font-family': 'monospace', 'padding-left': '0.2em', 'padding-right': '0.2em',
        'border-radius': '0 3px 0 0'
    }

    // create and add the button to all the code blocks that contain >>>
    div.each(function(index) {
        var jthis = $(this);
        if (jthis.find('.gp').length > 0) {
            var button = $('<span class="copybutton">&gt;&gt;&gt;</span>');
            button.css(button_styles)
            button.attr('title', hide_text);
            button.data('hidden', 'false');
            jthis.prepend(button);
        }
        // tracebacks (.gt) contain bare text elements that need to be
        // wrapped in a span to work with .nextUntil() (see later)
        jthis.find('pre:has(.gt)').contents().filter(function() {
            return ((this.nodeType == 3) && (this.data.trim().length > 0));
        }).wrap('<span>');
    });

    // define the behavior of the button when it's clicked
    $('.copybutton').click(function(e){
        e.preventDefault();
        var button = $(this);
        if (button.data('hidden') === 'false') {
            // hide the code output
            button.parent().find('.go, .gp, .gt').hide();
            button.next('pre').find('.gt').nextUntil('.gp, .go').css('visibility', 'hidden');
            button.css('text-decoration', 'line-through');
            button.attr('title', show_text);
            button.data('hidden', 'true');
        } else {
            // show the code output
            button.parent().find('.go, .gp, .gt').show();
            button.next('pre').find('.gt').nextUntil('.gp, .go').css('visibility', 'visible');
            button.css('text-decoration', 'none');
            button.attr('title', hide_text);
            button.data('hidden', 'false');
        }
    });
});
