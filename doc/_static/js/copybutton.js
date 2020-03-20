/*
 * MIT License
 *
 * Copyright (c) 2020 Jonathan Bac
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
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
