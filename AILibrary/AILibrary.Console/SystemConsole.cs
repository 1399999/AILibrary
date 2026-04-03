namespace AILibrary;

public class SystemConsole
{
    /// <summary>
    /// Turns the console into a selection of options (more simple).
    /// </summary>
    /// <param name="options">The parmaters which the users can choose from (in list form).</param>
    /// <returns>The index which the user selects.</returns>
    public static int ConvertIntoOptionsMode(List<string> options, string? topText = null)
    {
        int currentIndex = 0;

        Console.CursorVisible = false;

        while (true)
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.BackgroundColor = ConsoleColor.Black;

            Console.Clear();

            if (topText != null)
            {
                Console.WriteLine(topText + "\n");
            }

            for (int i = 0; i < options.Count; i++)
            {
                Console.ForegroundColor = ConsoleColor.White;
                Console.BackgroundColor = ConsoleColor.Black;

                if (currentIndex == i)
                {
                    Console.ForegroundColor = ConsoleColor.Black;
                    Console.BackgroundColor = ConsoleColor.White;
                }

                Console.WriteLine(options[i]);
            }

            var key = Console.ReadKey().Key;

            if (key == ConsoleKey.DownArrow)
            {
                currentIndex = currentIndex < options.Count - 1 ? currentIndex + 1 : 0;
                continue;
            }

            else if (key == ConsoleKey.UpArrow)
            {
                currentIndex = currentIndex - 1 >= 0 ? currentIndex - 1 : options.Count - 1;
                continue;
            }

            else if (key == ConsoleKey.Enter)
            {
                Console.ForegroundColor = ConsoleColor.White;
                Console.BackgroundColor = ConsoleColor.Black;

                return currentIndex;
            }

            else
            {
                continue;
            }
        }
    }

    public static void DisplayCredits() => ConvertIntoOptionsMode(["Exit"], "(C) Mikhail Zhebrunov 2025-2026\n\nA realitively lightwight AI library on CPU with no optimization");
}
